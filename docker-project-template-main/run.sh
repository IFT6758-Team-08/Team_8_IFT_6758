# #!/bin/bash

# echo "TODO: fill in the docker run command"
docker run -it -e 127.0.0.1:5000:5000/tcp --env COMET_API_KEY=$COMET_API_KEY ift6758/serving:$1
docker run -it -e 127.0.0.1:8501:8501/tcp --env ift6758/serving:$1