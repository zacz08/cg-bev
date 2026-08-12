import os
import argparse
import torch
from cldm.model import create_model, load_state_dict

def get_node_name(name, parent_name):
    if len(name) <= len(parent_name):
        return False, ''
    p = name[:len(parent_name)]
    if p != parent_name:
        return False, ''
    return True, name[len(parent_name):]


def save_state_dict_txt(state_dict, output_path):
    with open(output_path, 'w') as f:
        for k in state_dict.keys():
            f.write(f'{k}\n')


def _same_shape(source, target):
    return hasattr(source, 'shape') and hasattr(target, 'shape') and source.shape == target.shape


def build_cldm_state_dict_from_ldm(model, sd_weight_path,
                                   location='cpu', verbose=True,
                                   max_new_key_logs=20):
    pretrained_weights = load_state_dict(sd_weight_path, location=location)

    scratch_dict = model.state_dict()
    target_dict = {}
    copied = 0
    newly_added = []
    shape_mismatched = []

    for k, scratch_value in scratch_dict.items():
        is_control, name = get_node_name(k, 'control_model.')
        copy_k = 'model.unet.' + name if is_control else k

        if copy_k in pretrained_weights and _same_shape(pretrained_weights[copy_k], scratch_value):
            target_dict[k] = pretrained_weights[copy_k].clone()
            copied += 1
        else:
            target_dict[k] = scratch_value.clone()
            if copy_k in pretrained_weights:
                shape_mismatched.append((k, copy_k, tuple(pretrained_weights[copy_k].shape), tuple(scratch_value.shape)))
            else:
                newly_added.append(k)

    if verbose:
        print(f'[Init] copied {copied} tensors from LDM into CLDM skeleton.')
        for k in newly_added[:max_new_key_logs]:
            print(f'[Init] newly added CLDM tensor: {k}')
        if len(newly_added) > max_new_key_logs:
            print(f'[Init] ... {len(newly_added) - max_new_key_logs} more newly added tensors not shown.')
        for k, copy_k, src_shape, dst_shape in shape_mismatched[:max_new_key_logs]:
            print(f'[Init] shape mismatch, kept scratch tensor: {k} <- {copy_k} {src_shape} != {dst_shape}')
        if len(shape_mismatched) > max_new_key_logs:
            print(f'[Init] ... {len(shape_mismatched) - max_new_key_logs} more shape mismatches not shown.')

    return target_dict


def init_cldm_from_ldm(model, sd_weight_path, location='cpu', verbose=True):
    target_dict = build_cldm_state_dict_from_ldm(
        model,
        sd_weight_path=sd_weight_path,
        location=location,
        verbose=verbose,
    )
    model.load_state_dict(target_dict, strict=True)
    return model


def create_cldm_checkpoint_from_ldm(config_path, sd_weight_path, output_path,
                                    location='cpu'):
    model = create_model(config_path=config_path)
    init_cldm_from_ldm(
        model,
        sd_weight_path=sd_weight_path,
        location=location,
        verbose=True,
    )
    torch.save(model.state_dict(), output_path)
    print('Done.')
    print(f'Saved the new state_dict to [{output_path}]')


def main():
    create_cldm_checkpoint_from_ldm(
        config_path=config_path,
        sd_weight_path=sd_weight_path,
        output_path=output_path,
    )

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Initialise a CLDM checkpoint from an LDM checkpoint.')
    parser.add_argument('sd_weight_path', help='Path to source LDM .ckpt')
    parser.add_argument('output_path', help='Path to write initialised CLDM .ckpt')
    parser.add_argument('--config', default='./configs/cldm_res_192.yaml',
                        help='CLDM config to instantiate the target skeleton.')
    args = parser.parse_args()

    sd_weight_path = args.sd_weight_path
    output_path = args.output_path
    config_path = args.config

    assert os.path.exists(sd_weight_path), 'Input sd weight does not exist.'
    assert not os.path.exists(output_path), 'Output filename already exists.'
    assert os.path.exists(os.path.dirname(output_path)), 'Output path is not valid.'

    print(f'[Init] config        = {config_path}')
    print(f'[Init] LDM ckpt      = {sd_weight_path}')
    print(f'[Init] output ckpt   = {output_path}')
    main()
