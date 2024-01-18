from gyraudio.default_locations import EXPERIMENT_STORAGE_ROOT
from gyraudio.audio_separation.parser import shared_parser
from gyraudio.audio_separation.infer import launch_infer, RECORD_KEYS, SNR_OUT, SNR_IN, NBATCH, SAVE_IDX
from gyraudio.audio_separation.properties import TEST, NAME, SHORT_NAME, CURRENT_EPOCH, SNR_FILTER 
import sys
import os
from dash import Dash, html, dcc, callback, Output, Input, dash_table
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
from typing import List
import torch
from pathlib import Path
DIFF_SNR = 'SNR out - SNR in'



def get_app(record_row_dfs : pd.DataFrame, eval_dfs : List[pd.DataFrame]) :
    app = Dash(__name__)
    # names_options = [{'label' : f"{record[SHORT_NAME]} - {record[NAME]} epoch {record[CURRENT_EPOCH]:04d}", 'value' : record[NAME] } for idx, record in record_row_dfs.iterrows()]
    app.layout = html.Div([
        html.H1(children='Inference results', style={'textAlign':'center'}),
        # dcc.Dropdown(names_options, names_options[0]['value'], id='exp-selection'),
        # dcc.RadioItems(['scatter', 'box'], 'box', inline=True, id='radio-plot-type'),
        dcc.RadioItems([SNR_OUT, DIFF_SNR], DIFF_SNR, inline=True, id='radio-plot-out'),
        dcc.Graph(id='graph-content')
    ])

    @callback(
        Output('graph-content', 'figure'),
        # Input('exp-selection', 'value'),
        # Input('radio-plot-type', 'value'),
        Input('radio-plot-out', 'value'),
    )
    def update_graph(radio_plot_out) :
        fig = make_subplots(rows = 2, cols = 1)
        colors = px.colors.qualitative.Plotly
        for id, record in record_row_dfs.iterrows() :
            color = colors[id % len(colors)]
            eval_df = eval_dfs[id].sort_values(by=SNR_IN)
            eval_df[DIFF_SNR] = eval_df[SNR_OUT] - eval_df[SNR_IN]
            legend = f'{record[SHORT_NAME]}_{record[NAME]}'
            fig.add_trace(
                go.Scatter(
                    x=eval_df[SNR_IN], 
                    y=eval_df[radio_plot_out], 
                    mode="markers", marker={'color' : color}, 
                    name=legend,
                    hovertemplate = 'File : %{text}'+
                        '<br>%{y}<br>',
                    text = [f"{eval[SAVE_IDX]:.0f}" for idx, eval in eval_df.iterrows()]
                    ),
                row = 1, col = 1
            )
            eval_df_bins = eval_df
            eval_df_bins[SNR_IN] = eval_df_bins[SNR_IN].apply(lambda snr : round(snr))
            fig.add_trace(
                go.Box(x=eval_df[SNR_IN], y=eval_df[radio_plot_out], fillcolor = color, marker={'color' : color}, showlegend=False),
                row = 2, col = 1
            )

        title = f"SNR performances"
        fig.update_layout(
            title=title,
            xaxis2_title = SNR_IN,
            yaxis_title = radio_plot_out,
            hovermode='x unified'
            )
        return fig

        
    
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