# Submission Dockerfile for the ECML 2026 Flatland Competition.
#
# Uses the official flatland-baselines base image (which already ships flatland-rl
# and the conda env), then copies our submission folder and registers the
# observation builder + policy via environment variables.
#
# NOTE: The starterkit's main-branch Dockerfile says `COPY submission/ submission/`
#       but the repo actually contains a `my_orga/` folder. This file matches the
#       actual folder name. If the upstream repo standardizes on `submission/`,
#       rename `my_orga/` -> `submission/` and update the two POLICY/OBS_BUILDER
#       env vars below accordingly.
# https://docs.docker.com/reference/build-checks/invalid-default-arg-in-from/

ARG TAG=v4.2.5
FROM ghcr.io/flatland-association/flatland-baselines:${TAG}

COPY submission/ submission/

ENV POLICY=submission.my_policy.MyPolicy
ENV OBS_BUILDER=submission.my_observation_builder.MyObservationBuilderV3

RUN bash entrypoint_generic.sh python -m pip install -r submission/requirements.txt

