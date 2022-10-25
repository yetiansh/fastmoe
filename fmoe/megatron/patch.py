r"""
Patching some of Megatron-LM's functions to create an MoE model
"""
import os
import torch


def patch_forward_step(forward_step_func):
    r"""
    Patch model's forward_step_func to support balance loss
    """

    from megatron.mpu import is_pipeline_last_stage
    from megatron.mpu import get_tensor_model_parallel_group
    from megatron import get_args

    if not get_args().balance_strategy:
        return forward_step_func

    def forward_step_with_balance_loss(data_iterator, model, input_tensor):
        args = get_args()
        output = forward_step_func(data_iterator, model, input_tensor)

        moe_expert_interval = int(os.getenv("MOE_EXPERT_INTERVAL", 1))

        if not is_pipeline_last_stage() or not args.balance_strategy:
            return output

        while hasattr(model, 'module'):
            model = model.module

        if hasattr(model, "language_model"):
            loss_list = []
            if model.language_model.encoder:
                for idx, l in enumerate(model.language_model.encoder.layers):
                    if idx % moe_expert_interval == 0 and l.mlp.gate.has_loss:
                        loss_list.append(l.mlp.gate.get_loss(clear=False).view(1))

            if model.language_model.decoder:
                for idx, l in enumerate(model.language_model.decoder.layers):
                    if idx % moe_expert_interval == 0 and l.mlp.gate.has_loss:
                        loss_list.append(l.mlp.gate.get_loss(clear=False).view(1))
        else:
            loss_list = [
                loss_list.append(l.mlp.gate.get_loss(clear=False).view(1))
                for idx, l in enumerate(model.language_model.encoder.layers)
                if idx % moe_expert_interval == 0 and l.mlp.gate.has_loss
            ]

        if len(loss_list) == 0:
            return output

        loss_name = args.balance_strategy + "_loss"
        (loss, state_dict), bal_loss = (
            output,
            torch.cat(loss_list).mean() * args.balance_loss_weight
        )

        # avarage across moe group
        moe_group = get_tensor_model_parallel_group()
        world_size = torch.distributed.get_world_size(group=moe_group)
        averaged_bal_loss = bal_loss.clone().detach()
        torch.distributed.all_reduce(averaged_bal_loss, group=moe_group)
        averaged_bal_loss /= world_size

        loss += bal_loss
        state_dict[loss_name] = averaged_bal_loss

        return loss, state_dict

    return forward_step_with_balance_loss


def patch_model_provider(model_provider, gate=None):
    from megatron import get_args

    def fmoefied_model_provider(
        pre_process=None,
        post_process=None,
        add_encoder=None,
        add_decoder=None,
    ):
        from .layers import fmoefy
        args = get_args()
        hhs = args.hidden_size * 4
        assert hhs % args.top_k == 0
        hhs = hhs // args.top_k
        assert hhs % args.tensor_model_parallel_size == 0
        hhs = hhs // args.tensor_model_parallel_size
        kwargs_ = {
            "pre_process": pre_process,
            "post_process": post_process,
            "add_encoder": add_encoder,
            "add_decoder": add_decoder,
        }
        keys_to_delete = []
        for key, val in kwargs_.items():
            if val is None:
                keys_to_delete.append(key)
        for key in keys_to_delete:
            del kwargs_[key]
        return fmoefy(
            model_provider(
                **kwargs_
            ),
            num_experts=args.num_experts,
            hidden_hidden_size=hhs,
            top_k=args.top_k,
            gate=gate
        )

    return fmoefied_model_provider
