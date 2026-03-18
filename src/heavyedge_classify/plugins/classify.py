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

        from heavyedge_classify.model import minirocket_classifier

        self.logger.info(f"Writing {args.output}")

        with ProfileData(args.profiles) as file:
            X, _, _ = file[:]
        y = np.load(args.labels)

        model = minirocket_classifier(
            n_splits=args.n_splits,
            verbose=True,
            random_state=args.random_state,
        )
        model.fit(X, y)

        with open(args.output, "wb") as f:
            pickle.dump(model, f)

        self.logger.info(f"Saved {args.output}.")
