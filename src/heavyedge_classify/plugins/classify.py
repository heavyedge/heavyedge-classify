"""Commands for edge classification."""

import pathlib

from heavyedge.cli.command import Command, register_command

PLUGIN_ORDER = 1.0


@register_command("classify-train", "Train edge classifier")
class ClassifyTrainCommand(Command):
    def add_parser(self, main_parser):
        classify = main_parser.add_parser(
            self.name,
            description="Train edge classifier.",
            epilog="The output is a pkl file of the trained model.",
        )
        classify.add_argument(
            "profiles",
            type=pathlib.Path,
            help="Path to preprocessed profile data in 'ProfileData' structure.",
        )
        classify.add_argument(
            "labels",
            type=pathlib.Path,
            help=(
                "Path to label npy file. "
                "The order of labels should match the order of profiles."
            ),
        )
        classify.add_argument(
            "--n-splits",
            type=int,
            default=5,
            help="Number of splits for cross-validation (default=5).",
        )
        classify.add_argument(
            "--normalized",
            action="store_true",
            help=(
                "If input profiles are already normalized, "
                "setting this flag enhances performance."
            ),
        )
        classify.add_argument(
            "--random-state",
            type=int,
            default=0,
            help="Random seed for reproducibility (default=0).",
        )
        classify.add_argument(
            "-o", "--output", type=pathlib.Path, help="Output file path"
        )

    def run(self, args):
        import pickle

        import numpy as np
        from heavyedge.io import ProfileData

        from heavyedge_classify.api import classify_train

        self.logger.info(f"Training {args.output}")

        profiles = ProfileData(args.profiles)
        labels = np.load(args.labels)

        model = classify_train(
            profiles,
            labels,
            n_splits=args.n_splits,
            normalize=not args.normalized,
            random_state=args.random_state,
            logger=self.logger.info,
        )

        with open(args.output, "wb") as f:
            pickle.dump(model, f)

        self.logger.info(f"Saved {args.output}.")


@register_command("classify-predict", "Predict edge labels")
class ClassifyPredictCommand(Command):
    def add_parser(self, main_parser):
        classify = main_parser.add_parser(
            self.name,
            description="Predict probabilistic labels of edges using a trained model.",
            epilog=(
                "The output can be in npy (default) or csv format. ",
                "If csv, the first row is the header with class names. ",
                "Unrecognized formats are saved as npy with a warning.",
            ),
        )
        classify.add_argument(
            "profiles",
            type=pathlib.Path,
            help="h5 file path to profile data in 'ProfileData' structure.",
        )
        classify.add_argument(
            "model",
            type=pathlib.Path,
            help="Path to the trained model pkl file.",
        )
        classify.add_argument(
            "--normalized",
            action="store_true",
            help=(
                "If input profiles are already normalized, "
                "setting this flag enhances performance."
            ),
        )
        classify.add_argument(
            "--batch-size",
            type=int,
            default=None,
            help=(
                "Batch size to load data. "
                "If not passed, all data are loaded at once."
            ),
        )
        classify.add_argument(
            "--output-format",
            choices=["npy", "csv"],
            help="Output file format. If not passed, parsed from file extension.",
        )
        classify.add_argument(
            "-o", "--output", type=pathlib.Path, help="Output file path"
        )

    def run(self, args):
        import os
        import pickle

        from heavyedge.io import ProfileData

        from heavyedge_classify.api import classify_predict

        file_ext = os.path.splitext(args.output)[1].lower().lstrip(".")
        save_format = args.output_format or file_ext

        self.logger.info(f"Predicting {args.output}")

        with open(args.model, "rb") as f:
            model = pickle.load(f)

        profiles = ProfileData(args.profiles)

        generator = classify_predict(
            model,
            profiles,
            normalize=not args.normalized,
            batch_size=args.batch_size,
            logger=lambda msg: self.logger.info(f"{args.output} : {msg}"),
        )

        if save_format == "csv":
            import csv

            header = model.classes_
            with open(args.output, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(header)
                for probs in generator:
                    writer.writerows(probs)

        else:
            if save_format != "npy":
                self.logger.warning(
                    f"Unrecognized format '{save_format}', saving as npy."
                )

            import numpy as np

            probs = np.concatenate(list(generator), axis=0)
            np.save(args.output, probs)

        self.logger.info(f"Saved {args.output}.")
