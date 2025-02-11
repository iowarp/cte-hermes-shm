VERSION=v2.0.1
docker build -t grc-spack docker -f docker/grc-spack.Dockerfile
docker tag grc-spack lukemartinlogan/grc-repo:${VERSION}
docker push lukemartinlogan/grc-repo:${VERSION}
