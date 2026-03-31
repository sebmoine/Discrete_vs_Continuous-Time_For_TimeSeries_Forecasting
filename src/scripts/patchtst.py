from src.utils.exp.exp_long_term_forecasting import Exp_Main
import numpy as np
from argparse import Namespace
import torch
import random


def patchtst(cfg):
    """
        source : official paper's github PatchTST : https://github.com/yuqinie98/PatchTST/tree/main
    """
    seed=cfg["SEED"]
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)


    for pred_len in cfg["pred_len"]:
        data=cfg["data"]

        args = Namespace(
            seed=seed,
            pred_len=pred_len,
            data=data,
            root_path=cfg["root_path"],
            data_path=cfg["data_path"],
            checkpoints=cfg["logging"]["checkpoints"],
            results=cfg["logging"]["scores"],
            predictions=cfg["logging"]["predictions"],

            model_id = f"{data}_{pred_len}",
            model='PatchTST',
            features=cfg["features"],
            target="OT",
            freq='h',
            is_training=cfg["is_training"],
            seq_len=cfg["seq_len"],
            label_len=cfg["label_len"],

            # PatchTST
            fc_dropout=cfg["fc_dropout"],
            head_dropout=cfg["head_dropout"],
            patch_len=cfg["patch_len"],
            stride=cfg["stride"],
            padding_patch=cfg["padding_patch"],
            revin=cfg["revin"],
            affine=cfg["affine"],
            subtract_last=cfg["subtract_last"],
            decomposition=cfg["decomposition"],
            kernel_size=cfg["kernel_size"],
            individual=cfg["individual"],

            # Formers
            enc_in=cfg["enc_in"],
            e_layers=cfg["e_layers"],
            n_heads=cfg["n_heads"],
            d_model=cfg["d_model"],
            d_ff=cfg["d_ff"],
            dropout=cfg["dropout"],
            embed_type=cfg["embed_type"],
            dec_in=cfg["dec_in"],
            c_out=cfg["c_out"],
            d_layers=cfg["d_layers"],
            moving_avg=cfg["moving_avg"],
            factor=cfg["factor"],
            distil=cfg["distil"],
            embed=cfg["embed"],
            activation=cfg["activation"],
            output_attention=cfg["output_attention"],
            do_predict=cfg["do_predict"],

            # optimization
            des=cfg["des"],
            train_epochs=cfg["train_epochs"],
            itr=cfg["itr"],
            batch_size=cfg["batch_size"],
            learning_rate=cfg["learning_rate"],
            num_workers=cfg["num_workers"],
            patience=cfg["patience"],
            loss=cfg["loss"],
            lradj=cfg["lradj"],
            pct_start=cfg["pct_start"],
            use_amp=cfg["use_amp"],

            # GPU
            use_gpu=cfg["use_gpu"],
            gpu=cfg["gpu"],
            use_multi_gpu=cfg["use_multi_gpu"],
            devices=cfg["devices"],
            test_flop=cfg["test_flop"]
            )

        args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

        if args.use_gpu and args.use_multi_gpu:
            args.dvices = args.devices.replace(' ', '')
            device_ids = args.devices.split(',')
            args.device_ids = [int(id_) for id_ in device_ids]
            args.gpu = args.device_ids[0]

        task_name = 'long_term_forecast'
        print('Long Term Forecast')
        Exp = Exp_Main
        # print('Exp:', Exp)


        if args.is_training:
            for ii in range(args.itr):
                # setting record of experiments
                setting = '{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_eb{}_dt{}_{}_{}'.format(
                        args.model_id,
                        args.model,
                        args.data,
                        args.features,
                        args.seq_len,
                        args.label_len,
                        args.pred_len,
                        args.d_model,
                        args.n_heads,
                        args.e_layers,
                        args.d_layers,
                        args.d_ff,
                        args.factor,
                        args.embed,
                        args.distil,
                        args.des,
                        ii)

                exp = Exp(args)  # set experiments
                print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
                exp.train(setting)

                print('>>>>>>>validating : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
                exp.test(setting)

                if args.do_predict:
                    print('>>>>>>>predicting : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
                    exp.predict(setting, True)

                torch.cuda.empty_cache()
        else:
            ii = 0
            setting = '{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_eb{}_dt{}_{}_{}'.format(args.model_id,
                                                                                                            args.model,
                                                                                                            args.data,
                                                                                                            args.features,
                                                                                                            args.seq_len,
                                                                                                            args.label_len,
                                                                                                            args.pred_len,
                                                                                                            args.d_model,
                                                                                                            args.n_heads,
                                                                                                            args.e_layers,
                                                                                                            args.d_layers,
                                                                                                            args.d_ff,
                                                                                                            args.factor,
                                                                                                            args.embed,
                                                                                                            args.distil,
                                                                                                            args.des,
                                                                                                            ii)

            exp = Exp(args)  # set experiments
            print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
            exp.test(setting, test=1)
            torch.cuda.empty_cache()