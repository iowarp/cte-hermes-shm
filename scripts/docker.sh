docker build -t grc-spack docker -f docker/grc-spack.Dockerfile
docker tag grc-spack lukemartinlogan/grc-repo:v2.0.0
docker push lukemartinlogan/grc-repo:v2.0.0
