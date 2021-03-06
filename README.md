# MLDockerImage
A Docker Image for running Tensorflow-gpu on Docker.

This Requires installing nvidia-docker2 on the system and using the nvidia runtime so as to enable the visibility of the GPU to the container instance.

To use docker-compose to directly use the nvidia runtime by default set the ```"default-runtime"```  as ```"nvidia"``` in /etc/docker/daemon.json.

The base image is derived from nvidia/cuda:9.0-cudnn7-runtime. Replace with appropriate version for the specific usecase from [here](https://gitlab.com/nvidia/cuda). 


```bash
sudo apt autoremove
sudo apt-get installsudo apt-get install     apt-transport-https     ca-certificates     curl     software-properties-common
sudo apt-get install     apt-transport-https     ca-certificates     curl     software-properties-common
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo apt-key add -
sudo apt-key fingerprint 0EBFCD88
sudo add-apt-repository    "deb [arch=amd64] https://download.docker.com/linux/ubuntu \
   $(lsb_release -cs) \
   stable"
sudo apt-get update
sudo apt-get install docker-ce
sudo docker ps
sudo docker ps -a
sudo docker -v

# If you have nvidia-docker 1.0 installed: we need to remove it and all existing GPU containers
docker volume ls -q -f driver=nvidia-docker | xargs -r -I{} -n1 docker ps -q -a -f volume={} | xargs -r docker rm -f
sudo apt-get purge -y nvidia-docker
sudo ls
# If you have nvidia-docker 1.0 installed: we need to remove it and all existing GPU containers
docker volume ls -q -f driver=nvidia-docker | xargs -r -I{} -n1 docker ps -q -a -f volume={} | xargs -r docker rm -f
sudo apt-get purge -y nvidia-docker
# Add the package repositories
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey |   sudo apt-key add -
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list |   sudo tee /etc/apt/sources.list.d/nvidia-docker.list
sudo apt-get update
# Install nvidia-docker2 and reload the Docker daemon configuration
sudo apt-get install -y nvidia-docker2
sudo pkill -SIGHUP dockerd
# Test nvidia-smi with the latest official CUDA image
sudo docker run --runtime=nvidia --rm nvidia/cuda:9.0-base nvidia-smi

sudo apt install docker-compose
sudo docker-compose build
sudo docker-compose up
```