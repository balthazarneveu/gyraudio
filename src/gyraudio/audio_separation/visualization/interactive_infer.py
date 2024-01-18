from gyraudio.default_locations import EXPERIMENT_STORAGE_ROOT
from gyraudio.audio_separation.parser import shared_parser
from gyraudio.audio_separation.infer import launch_infer, RECORD_KEYS, SNR_OUT, SNR_IN, NBATCH
from gyraudio.audio_separation.properties import TEST, NAME, SHORT_NAME, CURRENT_EPOCH, SNR_FILTER 
import sys
import os
from dash import Dash, html, dcc, callback, Output, Input, dash_table
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from typing import List
import torch
from pathlib import Path
DIFF_SNR = 'SNR out - SNR in'



def get_app(record_row_dfs : pd.DataFrame, eval_dfs : List[pd.DataFrame]) :
    app = Dash(__name__)
    names_options = [{'label' : f"{record[SHORT_NAME]} - {record[NAME]} epoch {record[CURRENT_EPOCH]:04d}", 'value' : record[NAME] } for idx, record in record_row_dfs.iterrows()]
    app.layout = html.Div([
        html.H1(children='Inference results', style={'textAlign':'center'}),
        dcc.Dropdown(names_options, names_options[0]['value'], id='exp-selection'),
        dcc.RadioItems(['scatter', 'box'], 'box', inline=True, id='radio-plot-type'),
        dcc.RadioItems([SNR_OUT, DIFF_SNR], DIFF_SNR, inline=True, id='radio-plot-out'),
        dcc.Graph(id='graph-content')
    ])

    @callback(
        Output('graph-content', 'figure'),
        Input('exp-selection', 'value'),
        Input('radio-plot-type', 'value'),
        Input('radio-plot-out', 'value'),
    )
    def update_graph(exp_selection, radio_plot_type, radio_plot_out):
        id = (record_row_dfs.name == exp_selection).idxmax()
        record = record_row_dfs.loc[id]
        eval_df = eval_dfs[id].sort_values(by=SNR_IN)
        eval_df[DIFF_SNR] = eval_df[SNR_OUT] - eval_df[SNR_IN]

        title = f"SNR performance for experience {record[NAME]} in epoch {record[CURRENT_EPOCH]:04d} with {len(eval_df)} samples"
        fig = px.scatter(eval_df, x=SNR_IN, y=radio_plot_out, title=title, trendline='ols') #, marginal_y = "histogram"
        fig.data[0].name = "Inference data"
        fig.data[0].showlegend = True
        model = px.get_trendline_results(fig)
        results = model.iloc[0]["px_fit_results"]
        alpha = results.params[0]
        beta = results.params[1]
        fig.data[1].name = f"LS regression {radio_plot_out} = {alpha:.2f} + {beta:.2f} {SNR_IN}"
        fig.data[1].showlegend = True


        eval_df[SNR_IN] = eval_df[SNR_IN].apply(lambda snr : round(snr))
        fig2 = px.box(eval_df, x=SNR_IN, y=radio_plot_out, title=title) # ,points="all"
        # fig.add_trace(
        #     go.Scatter(
        #         x=eval_df[SNR_IN],
        #         y=eval_df[SNR_IN],
        #         mode="lines",
        #         line=go.scatter.Line(color="gray"),
        #         )
        # ) 
        if radio_plot_type == 'scatter' :
            return fig
        return fig2

        
    
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
    record_row_dfs = pd.DataFrame(columns = RECORD_KEYS)
    eval_dfs = []
    for exp in args.experiments:
        record_row_df, evaluation_path = launch_infer(
                exp,
                model_dir=Path(args.input_dir),
                output_dir=Path(args.output_dir),
                device=args.device,
                max_batches=args.nb_batch,
                snr_filter=args.snr_filter
            )
        eval_df = pd.read_csv(evaluation_path)
        # Careful, list order for concat is important for index matching eval_dfs list
        record_row_dfs = pd.concat([record_row_dfs.loc[:], record_row_df], ignore_index=True)
        eval_dfs.append(eval_df)
    app = get_app(record_row_dfs, eval_dfs)
    app.run(debug=True)


if __name__ == '__main__':
    os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
    main(sys.argv[1:])