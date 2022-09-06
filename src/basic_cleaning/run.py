#!/usr/bin/env python
"""
Download from W&B the raw dataset and apply some basic data cleaning, exporting the result to a new artifact
"""
import argparse
import logging
import os
import tempfile

import pandas as pd
import wandb


logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


def go(args):

    run = wandb.init(job_type="basic_cleaning")
    run.config.update(args)

    # Download input artifact. This will also log that this script is using this
    # particular version of the artifact
    # artifact_local_path = run.use_artifact(args.input_artifact).file()

    ######################
    # YOUR CODE HERE     #
    ######################
    logger.info(f"Fetching dataset '{args.input_artifact}'")
    artifact_local_path = run.use_artifact(args.input_artifact).file()
    df = pd.read_csv(artifact_local_path)

    # Drop outliers
    logger.info(f"Dropping outliers")
    df = df[
        df['price'].between(args.min_price, args.max_price)
    ].copy()
    # Convert last_review to datetime
    logger.info(f"Converting datetime columns to datetime type")
    df['last_review'] = pd.to_datetime(df['last_review'])

    # Drop outlier geolocation coords
    idx = df['longitude'].between(-74.25, -73.50) & df['latitude'].between(40.5, 41.2)
    df = df[idx].copy()

    # Save the artifacts. We use a temporary directory so we do not leave
    # any trace behind
    with tempfile.TemporaryDirectory() as tmp_dir:

        logger.info("Saving to temp folder localy before upload")

        # Get the path on disk within the temp directory
        temp_path = os.path.join(tmp_dir, args.output_artifact)

        # Save then upload to W&B
        df.to_csv(temp_path, index=False)

        artifact = wandb.Artifact(
            name=args.output_artifact,
            type=args.output_type,
            description=args.output_description,
        )
        artifact.add_file(temp_path)

        logger.info(f"Uploading {args.output_artifact} to Weights & Biases")
        run.log_artifact(artifact)

        # This waits for the artifact to be uploaded to W&B. If you
        # do not add this, the temp directory might be removed before
        # W&B had a chance to upload the datasets, and the upload
        # might fail
        artifact.wait()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="A very basic data cleaning")


    parser.add_argument(
        "--input_artifact", 
        type=str,
        help="Name of the input artifact of the previous step",
        required=True
    )

    parser.add_argument(
        "--output_artifact", 
        type=str,
        help="Name of the output artifact of the current step",
        required=True
    )

    parser.add_argument(
        "--output_type", 
        type=str,
        help="Type of the output artifact",
        required=True
    )

    parser.add_argument(
        "--output_description", 
        type=str,
        help="Description of the output artifact",
        required=True
    )

    parser.add_argument(
        "--min_price", 
        type=float,
        help="Min price to clip low outliers to",
        required=True
    )

    parser.add_argument(
        "--max_price", 
        type=float,
        help="Max price to clip high outliers to",
        required=True
    )


    args = parser.parse_args()

    go(args)
