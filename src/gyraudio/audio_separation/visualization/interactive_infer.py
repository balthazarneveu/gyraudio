from gyraudio.default_locations import EXPERIMENT_STORAGE_ROOT
from gyraudio.audio_separation.parser import shared_parser
from gyraudio.audio_separation.infer import launch_infer, already_inferred, DEFAULT_RECORD_FILE, DEFAULT_EVALUATION_FILE,  RECORD_KEYS, NBATCH, SNR_OUT, SNR_IN
from gyraudio.audio_separation.properties import TEST, NAME, SHORT_NAME, CURRENT_EPOCH, SNR_FILTER 
from gyraudio.audio_separation.experiment_tracking.storage import get_output_folder, load_checkpoint
from gyraudio.audio_separation.experiment_tracking.experiments import get_experience
import sys
from dash import Dash, html, dcc, callback, Output, Input, dash_table
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import torch
from pathlib import Path



def get_app(eval_df : pd.DataFrame) :
    eval_df = eval_df.sort_values(by=SNR_IN)

    app = Dash(__name__)
    fig = px.scatter(eval_df, x=SNR_IN, y=SNR_OUT, marginal_y="histogram")
    fig.add_trace(
        go.Scatter(
            x=eval_df[SNR_IN],
            y=eval_df[SNR_IN],
            mode="lines",
            line=go.scatter.Line(color="gray"),
            )
    )
    app.layout = html.Div([
        html.H1(children='Visualize validation', style={'textAlign':'center'}),
        dcc.Graph(id='graph-content', figure=fig),
        # dash_table.DataTable(data=eval_df.sort_values(by=save_idx).to_dict('records'), page_size=10),
    ])
    
    return app
    
def main(argv):
    default_device = "cuda" if torch.cuda.is_available() else "cpu"
    parser_def = shared_parser(help="Launch training \nCheck results at: https://wandb.ai/balthazarneveu/audio-sep"
                               + ("\n<<<Cuda available>>>" if default_device == "cuda" else ""))
    parser_def.add_argument("-i", "--input-dir", type=str, default=EXPERIMENT_STORAGE_ROOT)
    parser_def.add_argument("-o", "--output-dir", type=str, default=EXPERIMENT_STORAGE_ROOT)
    parser_def.add_argument("-d", "--device", type=str, default=default_device,
                            help="Training device", choices=["cpu", "cuda"])
    parser_def.add_argument("-b", "--nb-batch", type=int, default=None,
                    help="Number of batches to process")
    parser_def.add_argument("-s",  "--snr-filter", type=float, nargs="+", default=None,
                    help="SNR filters on the inference dataloader")
    args = parser_def.parse_args(argv)
    for exp in args.experiments:
        # load experience
        if args.snr_filter is not None :
            args.snr_filter = sorted(args.snr_filter)
        short_name, model, config, dl = get_experience(exp, snr_filter_test=args.snr_filter)
        exists, exp_dir = get_output_folder(config, root_dir=Path(args.input_dir), override=False)
        assert exp_dir.exists(), f"Experiment {short_name} does not exist in {Path(args.input_dir)}"
        model.eval()
        model.to(args.device)
        model, optimizer, epoch, config_checkpt = load_checkpoint(model, exp_dir, epoch=None, device=args.device)
        if args.output_dir is not None :
            record_path = Path(args.output_dir)/DEFAULT_RECORD_FILE
            assert record_path.exists()
            record_df = pd.read_csv(record_path)
            record_row = pd.DataFrame({
                NAME: config[NAME],
                SHORT_NAME: config[SHORT_NAME],
                CURRENT_EPOCH: epoch,
                NBATCH: args.nb_batch,
                SNR_FILTER: [None],
            }, index = [0], columns = RECORD_KEYS)
            record_row.at[0, SNR_FILTER] = args.snr_filter
            if not(already_inferred(record_df, record_row)) :
                launch_infer(
                    exp,
                    model_dir=Path(args.input_dir),
                    output_dir=Path(args.output_dir),
                    device=args.device,
                    force_reload=args.reload,
                    max_batches=args.nb_batch,
                    snr_filter=args.snr_filter
                )
            save_dir = Path(args.output_dir)/(exp_dir.name+"_infer"+ (f"_epoch_{epoch:04d}_nbatch_{args.nb_batch if args.nb_batch is not None else len(dl[TEST])}")
                        + ("" if args.snr_filter is None else f"_snrs_{'_'.join(map(str, args.snr_filter))}"))
            evaluation_path = save_dir/DEFAULT_EVALUATION_FILE
            eval_df = pd.read_csv(evaluation_path)

            app = get_app(eval_df)
            app.run(debug=True)


if __name__ == '__main__':
    main(sys.argv[1:])