# Contributing guidelines

​     

## Pull Request Checklist

Before sending your pull requests, make sure you followed this list.  

- Changes are consistent with the [Coding Style](https://github.com/DeepBaksuVision/YOLO9000/CONTRIBUTING.md#Python coding style).
- Run [Usability check](https://github.com/DeepBaksuVision/YOLO9000/CONTRIBUTING.md#Running Usability check).

​    

## How to become a contributor and submit your own code

​     

### Contributor License Agreements

None

​     

### Contributing code

If you have improvements to YOLO9000, send us your pull requests! For those
just getting started, Github has a [howto](https://help.github.com/articles/using-pull-requests/).



we recommend repository structure like below illustrations

so, contributor should be follow procedure like below

1. **Fork** Original Repository to your repository

2. set remote name of your repository is **origin** and remote name of Original Repository is **upstream** at your local machine

3. register your issue at Original Repository(Upstream)

4. Make branch in your local machine and your repository. **it's should be named according issue number**

   ex) issue name : [Darknet19 pretrain weight transfer learning](https://github.com/DeepBaksuVision/YOLO9000/issues/15) 

   ​      issue number : #15 

   ​      branch name : issue_15 

5. Build your code in local machine and your repository at branch named issue number

6. Register **Pull Request** at Origin Repository. request your code at your repository merge into  original repository

7. Sync your repository with original repository. refer [here](https://github.com/DeepBaksuVision/YOLO9000/CONTRIBUTING.md#Sync repository with original repository)

![cejjt](https://user-images.githubusercontent.com/13328380/51797213-a4d32e00-2242-11e9-8235-7aca865d77be.png)

​    

#### Sync repository with original repository

1. adding two remote at your local machine

   ```bash
   $ git add remote origin <your repository>
   $ git add remote upstream <original repository>
   ```

2. sanity check

   ```bash
   $ git remote -v
   >
   origin	https://github.com/ssaru/YOLO9000.git (fetch)
   origin	https://github.com/ssaru/YOLO9000.git (push)
   upstream	https://github.com/DeepBaksuVision/YOLO9000.git (fetch)
   upstream	https://github.com/DeepBaksuVision/YOLO9000.git (push)
   ```

3. sync repository with original repository

   ```bash
   $ git fetch upstream
   $ git checkout <your branch that you wanna merge with origin branch>
   $ git merge upstream/<branch name>
   
   ex)
   $ git fetch upstream
   $ git checkout dev
   $ git merge upstream/dev
   ```

​     

### Contribution guidelines and standards

None

​          

#### General guidelines and philosophy for contribution

None

​    

#### License

None

​    

#### Python coding style

Changes to YOLO9000 Python code should conform to [Google Python Style Guide](https://github.com/google/styleguide/blob/gh-pages/pyguide.md)

refer [Usability check](https://github.com/DeepBaksuVision/YOLO9000/CONTRIBUTING.md#Running Usability check)

​     

#### Running Usability check

There are way to run YOLO9000 Usability check

```bash
$ make test
```

​    

#### Running unit tests

None

​    

## Reference

[[1]. **CONTRIBUTING.md** of Tensorflow](https://github.com/tensorflow/tensorflow/blob/master/CONTRIBUTING.md)

[[2]. What is the difference between origin and upstream in github](https://outofmymemory.wordpress.com/2013/09/18/what-is-the-difference-between-origin-and-upstream-in-github/)

