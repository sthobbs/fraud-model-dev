
from validate_serving import model_serving, upload_scores, compare_scores
from utils import run_queries


def run():

    # Pass data through model serving dataflow job
    model_serving.run()

    # Upload scores to GCS and BQ
    upload_scores.run()

    # Combine scores from BQ and Dataflow
    run_queries.run("./validate_serving/sql/format")
    run_queries.run("./validate_serving/sql/combine")

    # Compare scores
    compare_scores.run()


if __name__ == '__main__':
    run()
