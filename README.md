# Docker-based Data Science Template for VS Code

- Python
- Docker
- Jupyter
- TensorFlow
- TensorBoard
- GPU Support

## Starting Jupyter Lab and TensorBoard

`$ docker/docker-forever.sh [--jupyter_port=####|8888] [--tensorboard_port=####|6006]`

## Running Python from VS Code

- Build docker image with "Build Image" command from the command palette.
- Run or debug, python interpreter in the docker will be used automatically to execute the active file.

## Misc

- TensorBoard log dir inside container is at `/app/.tensorboard`.
- Edit `JUPYTER_TOKEN` argument in `docker/Dockerfile` to set Jupyter access token
