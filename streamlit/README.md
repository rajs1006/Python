# Hello World with Cloud Code

"Hello World" is a simple Kubernetes application that contains
[Deployments](https://kubernetes.io/docs/concepts/workloads/controllers/deployment/) for a web server and a database, and corresponding
[Services](https://kubernetes.io/docs/concepts/services-networking/service/). The Deployment contains a
[Django-based](https://www.djangoproject.com/) web server that simply prints "Hello World".

----

## Table of Contents

### Cloud Code for Visual Studio Code

[Using the Command Line](#using-the-command-line)
    * [Skaffold](#using-skaffold)
    * [kubectl](#using-kubectl)

----

### Using the Command Line

As an alternative to using the Cloud Code extension, the application can be deployed to a cluster using standard command line tools

#### Skaffold

[Skaffold](https://github.com/GoogleContainerTools/skaffold) is a command line tool that can be used to build, push, and deploy your container images

```bash
skaffold run --default-repo=gcr.io/your-project-id-here/cloudcode
```

#### kubectl

[kubectl](https://kubernetes.io/docs/tasks/tools/install-kubectl/) is the official Kubernetes command line tool. It can be used to deploy Kubernetes manifests to your cluster, but images must be build seperately using another tool (for example, using the [Docker CLI](https://docs.docker.com/engine/reference/commandline/cli/))

-----|------
