import os

import jammy.io as jio


def resolve_result_folder(config, ckpt_path):
  sampler_name = config.sampling.method
  if sampler_name.lower() == 'order0':
    meta_info = {
        "tsOrder": config.sampling.ts_order,
        "nfe": config.sampling.nfe,
        "eps": config.sampling.t0,
    }
    info_path = f'order0'
    for key, value in meta_info.items():
        info_path += f"_{key}{value}"
    results_folder = os.path.join(f"{ckpt_path}_eval", info_path)
    jio.mkdir(results_folder)
    jio.dump(
        f"{results_folder}/meta.pkl",
        meta_info
    )
    return results_folder
  else:
    raise RuntimeError(f"{sampler_name} not supported")
