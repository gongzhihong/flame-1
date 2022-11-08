import yaml
import click
import torch
import shutil
import logging
import zipfile
from pathlib import Path
from collections import OrderedDict


@click.command()
@click.option('--directory', default='./ext_data', help='Directory with downloaded data')
@click.option('--cpu', is_flag=True, help='Model will be run on CPU only')
def main(directory, cpu):

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)-7.7s]  %(message)s",
        datefmt="%Y-%m-%d %H:%M",
        handlers=[
            logging.StreamHandler(),
        ]
    )
    logger = logging.getLogger("flame")
    
    cfg = {}
    
    data_dir = Path(directory).resolve()
    if not data_dir.is_dir():
        logger.exception(f"Directory '{directory}' does not exist!")
        exit()

    flame_model_path = data_dir / 'FLAME/generic_model.pkl'
    if flame_model_path.is_file():
        logger.info("FLAME data is already configured!")
        cfg['flame_path'] = str(flame_model_path)
    else:
        flame_zip = data_dir / 'FLAME2020.zip'
        if not flame_zip.is_file():
            logger.warning(f"File '{str(flame_zip)}' does not exist!")
        else:
            logger.info("Unzipping FLAME2020.zip file ...")
            with zipfile.ZipFile(flame_zip, 'r') as zip_ref:
                zip_ref.extractall(f'{directory}/FLAME/')

            logger.info("FLAME is downloaded and configured correctly!")
            cfg['flame_path'] = str(flame_model_path)

    deca_model_path = data_dir / 'deca_model.tar'
    if deca_model_path.is_file():
        logger.info("DECA model is already configured!")
        cfg['deca_path'] = str(deca_model_path)
    else:
        logger.warning(f'File {deca_model_path} does not exist!')

    device = 'cpu' if cpu else 'cuda'
    ckpt_out = data_dir / 'emoca.ckpt'
    if ckpt_out.is_file():
        logger.info("EMOCA model is already configured!")
        cfg['emoca_path'] = str(ckpt_out)
    else:
        logger.info(f"Configuring EMOCA for device '{device}'!")
        emoca_zip = data_dir / 'EMOCA.zip'
        if not emoca_zip.is_file():
            logger.warning(f"File '{str(emoca_zip)}' does not exist!")
        else:
            logger.info("Unzipping EMOCA.zip file ...")
            with zipfile.ZipFile(emoca_zip, 'r') as zip_ref:
                zip_ref.extractall(f'{directory}/')

            cfg = data_dir / 'EMOCA/cfg.yaml'
            if cfg.is_file():
                cfg.unlink()

            ckpt = list(data_dir.glob("**/*.ckpt"))
            if len(ckpt) == 0:
                logger.exception("Could not find EMOCA .ckpt file!", exc_info=False)
                exit()

            ckpt = ckpt[0]
            shutil.move(ckpt, data_dir / "emoca.ckpt")
            shutil.rmtree(data_dir / 'EMOCA')
            
            logger.info("Reorganizing EMOCA checkpoint file ... ")
            sd = torch.load(data_dir / "emoca.ckpt")["state_dict"]
            models = ["E_flame", "E_detail", "E_expression", "D_detail"]

            state_dict = {}
            for mod in models:
                state_dict[mod] = OrderedDict()

                for key, value in sd.items():
                    if mod in key:
                        k = key.split(mod + ".")[1]
                        state_dict[mod][k] = value.to(device=device)

            torch.save(state_dict, ckpt_out)
            logger.info(f"Saving reorganized checkpoint file at {str(ckpt_out)}")
            cfg['emoca_path'] = str(ckpt_out)
            
    mica_model_path = data_dir / 'mica.tar'
    if mica_model_path.is_file():
        logger.info("MICA model is already configured!")
        cfg['mica_path'] = str(mica_model_path)
    else:
        logger.warning(f'File {deca_model_path} does not exist!')

    spectre_model_path = data_dir / 'spectre_model.tar'
    if spectre_model_path.is_file():
        logging.info("Spectre model is already configured!")
        cfg['spectre_path'] = str(spectre_model_path)
    else:
        logger.warning(f'File {spectre_model_path} does not exist!')

    cfg_path = Path('./flame/data/config.yaml')
    with open(cfg_path, 'w') as f_out:
        logger.info(f"Saving config file to {cfg_path}!")
        yaml.dump(cfg, f_out, default_flow_style=False)


if __name__ == '__main__':
    main()