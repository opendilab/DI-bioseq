import os
import argparse

from bioseq.pipeline import Pipeline
from bioseq.landscape import create_landscape
from bioseq.engine import create_engine
from bioseq.model import create_model
from bioseq.encoder import create_encoder
from bioseq.utils.draw_results import draw_results


def main(args):
    score_per_round = args.score
    predict_per_round = args.predict
    landscape = create_landscape(args.landscape)
    codebook = landscape.codebook
    model = create_model(args.model)
    encoder = create_encoder(args.encoder, codebook)
    engine = create_engine(args.engine, model, encoder, codebook, predict_per_round, landscape.seq_len)

    rounds = args.round
    pipeline = Pipeline(
        landscape=landscape,
        engine=engine,
        rounds=rounds,
        score_per_round=score_per_round,
        predict_per_round=predict_per_round,
        log_dir='results',
    )

    for i, start_point in enumerate(landscape.starts):
        print("Start pipeline with seq: ", start_point)
        pipeline.reset()
        pipeline.run_pipeline(start_point, i)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--landscape', type=str, default='gb1:with_imputed', help="landscape type")
    parser.add_argument('--engine', type=str, default='random', help="searching engine type")
    parser.add_argument('--model', type=str, default='linear', help="prediction model type")
    parser.add_argument('--encoder', type=str, default='onehot', help="encoder type")
    parser.add_argument('--score', type=int, default=384, help="max score in landscape each round")
    parser.add_argument('--predict', type=int, default=3200, help="max prediction in model each round")
    parser.add_argument('--round', type=int, default=10, help="searching rounds")
    parser.add_argument('--logdir', type=str, default='results', help="result logging folder path")

    args = parser.parse_known_args()[0]

    main(args)
