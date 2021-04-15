1 git pull
2 git checkout -b my-branch //create branch locally
3 git add chemin/vers/mon/fichier.scala // git add . for adding all files
4 git commit -m "Mon message" 
5 git push --set-upstream origin my-branch // push the local branch to remote server / github server
6 git checkout origin/master // switch to the master branch 
7 git pull // get all the update
8 git checkout my-branch // switch to my branch
9 git rebase origin/master // combine the master branch and my branch, solve the conflicts, modifier or delete etc
10 git push –force // a merge request will appear on the github or gitlab
11 [merge on UI GitLab] // do merge on gitlab/github
12 git checkout master 
13 git branch --delete my-branch // delete my branch, all the modification has been applied to the master branch 
