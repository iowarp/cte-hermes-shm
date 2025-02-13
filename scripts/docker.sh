# docker login
docker build -t grc-spack docker -f docker/grc-spack.Dockerfile
docker tag grc-spack lukemartinlogan/grc-repo:latest
docker push lukemartinlogan/grc-repo:latest

VERSION=v2.0.1
docker tag grc-spack lukemartinlogan/grc-repo:${VERSION}
docker push lukemartinlogan/grc-repo:${VERSION}