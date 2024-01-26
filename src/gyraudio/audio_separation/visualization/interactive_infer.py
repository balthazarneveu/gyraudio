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



def get_app(record_row_dfs : pd.DataFrame, eval_dfs : List[pd.DataFrame]) :
    app = Dash(__name__)
    names = [f"{record[SHORT_NAME]} - {record[NAME]} epoch {record[CURRENT_EPOCH]:04d}" for idx, record in record_row_dfs.iterrows()]
    app.layout = html.Div([
        html.H1(children='Inference results', style={'textAlign':'center'}),
        dcc.Dropdown(names, names[0], id='dropdown-selection'),
        dcc.Graph(id='graph-content')
    ])
    # eval_df = eval_df.sort_values(by=SNR_IN)
    # app = Dash(__name__)
    # fig = px.scatter(eval_df, x=SNR_IN, y=SNR_OUT, marginal_y="histogram")
    # fig.add_trace(
    #     go.Scatter(
    #         x=eval_df[SNR_IN],
    #         y=eval_df[SNR_IN],
    #         mode="lines",
    #         line=go.scatter.Line(color="gray"),
    #         )
    # )
    # app.layout = html.Div([
    #     html.H1(children='Visualize validation', style={'textAlign':'center'}),
    #     dcc.Graph(id='graph-content', figure=fig),
    #     # dash_table.DataTable(data=eval_df.sort_values(by=save_idx).to_dict('records'), page_size=10),
    # ])

    @callback(
        Output('graph-content', 'figure'),
        Input('dropdown-selection', 'value')
    )
    def update_graph(value):
        id = (record_row_dfs.name == value).idxmax()
        record = record_row_dfs.loc[id]
        eval_df = eval_dfs[id].sort_values(by=SNR_IN)
        title = f"SNR performance for experience {record[NAME]} in epoch {record[CURRENT_EPOCH]:04d}"

        fig = px.scatter(eval_df, x=SNR_IN, y=SNR_OUT, title=title, marginal_y = "histogram", trendline='ols', trendline_color_override='darkblue')
        fig.data[0].name = "Inference data"
        fig.data[0].showlegend = True
        fig.data[1].name = "Least Square regression"
        fig.data[1].showlegend = True
        # fig.add_trace(
        #     go.Scatter(
        #         x=eval_df[SNR_IN],
        #         y=eval_df[SNR_IN],
        #         mode="lines",
        #         line=go.scatter.Line(color="gray"),
        #         )
        # ) 
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
        record_row_dfs = pd.concat([record_row_df, record_row_dfs.loc[:]], ignore_index=True)
        eval_dfs.append(eval_df)
    app = get_app(record_row_dfs, eval_dfs)
    app.run(debug=True)


if __name__ == '__main__':
    main(sys.argv[1:])