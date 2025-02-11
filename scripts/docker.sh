# docker login
VERSION=v2.0.1
docker build --no-cache -t grc-spack docker -f docker/grc-spack.Dockerfile
docker tag grc-spack lukemartinlogan/grc-repo:${VERSION}
docker push lukemartinlogan/grc-repo:${VERSION}
docker tag grc-spack lukemartinlogan/grc-repo:latest
docker push lukemartinlogan/grc-repo:latest
