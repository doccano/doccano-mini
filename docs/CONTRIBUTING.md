# Contributing

When contributing to this repository, please first discuss the change you wish to make via issue with the owners of this repository before making a change.

Please note we have a code of conduct, please follow it in all your interactions with the project.

## How to contribute

### Reporting Bugs

#### Before submitting a bug report

* Ensure the bug was not already reported by searching on GitHub under [Issues](https://github.com/doccano/doccano-mini/issues).
* [Open a new issue](https://github.com/doccano/doccano-mini/issues/new/choose) if you're unable to find an open one addressing the problem.
* Use the relevant bug report templates to create the issue.

#### How do I submit a good bug report?

Explain the problem and include additional details to help maintainers reproduce the problem:

* Use a clear and descriptive title for the issue to identify the problem.
* Describe the exact steps which reproduce the problem in as many details as possible.
* Provide specific examples to demonstrate the steps.
* Describe the behavior you observed after following the steps and point out what exactly is the problem with that behavior.
* Explain which behavior you expected to see instead and why.
* Include screenshots and animated GIFs which show you following the described steps and clearly demonstrate the problem.
* If the problem is related to performance or memory, include a CPU profile capture with your report.
* If the problem is related to network, include a network activity in Chrome/Firefox/Safari DevTools.
* If the problem wasn't triggered by a specific action, describe what you were doing before the problem happened and share more information using the guidelines below.

### Suggesting Enhancements

#### Before submitting an enhancement suggestion

* Ensure the suggestion was not already reported by searching on GitHub under [Issues](https://github.com/doccano/doccano-mini/issues).
* [Open a new issue](https://github.com/doccano/doccano-mini/issues/new/choose) if you're unable to find an open one addressing the suggestion.
* Use the relevant issue templates to create one.

#### How do I submit a good enhancement suggestion?

Explain the suggestion and include additional details to help developers understand it:

* Use a clear and descriptive title for the issue to identify the suggestion.
* Provide a step-by-step description of the suggested enhancement in as many details as possible.
* Provide specific examples to demonstrate the steps.
* Describe the current behavior and explain which behavior you expected to see instead and why.
* Include screenshots and animated GIFs which help you demonstrate the steps or point out the part of doccano-mini which the suggestion is related to.
* Explain why this enhancement would be useful to most users.
* List some other annotation tools or applications where this enhancement exists.
* Specify which version you're using.
* Specify the name and version of the OS you're using.

## development workflow

1. **Fork the project & clone it locally:** Click the "Fork" button in the header of the [GitHub repository](https://github.com/doccano/doccano-mini), creating a copy of `doccano-mini` in your GitHub account. To get a working copy on your local machine, you have to clone your fork. Click the "Clone or Download" button in the right-hand side bar, then append its output to the `git clone` command.

        $ git clone https://github.com/YOUR_USERNAME/doccano-mini.git

1. **Create an upstream remote and sync your local copy:** Connect your local copy to the original "upstream" repository by adding it as a remote.

        $ cd doccano-mini
        $ git remote add upstream https://github.com:doccano/doccano-mini.git

    You should now have two remotes: read/write-able `origin` points to your GitHub fork, and a read-only `upstream` points to the original repo. Be sure to [keep your fork in sync](https://help.github.com/en/articles/syncing-a-fork) with the original, reducing the likelihood of merge conflicts later on.

1. **Create a branch for each piece of work:** Branch off `develop` for each bugfix or feature that you're working on. Give your branch a descriptive, meaningful name like `bugfix-for-issue-1234` or `improve-io-performance`, so others know at a glance what you're working on.

        $ git checkout develop
        $ git pull develop master && git push origin develop
        $ git checkout -b my-descriptive-branch-name

    At this point, you may want to install your version. It's usually best to do this within a dedicated virtual environment; We recomment to use `poetry`:

        $ poetry install
        $ poetry shell

    Then run the `streamlit` command to serve:

        $ streamlit run doccano_mini/app.py

    Now, you can access to the frontend at <http://localhost:8501/>.

2. **Implement your changes:** Use your preferred text editor to modify the source code. Be sure to keep your changes focused and in scope, and follow the coding conventions described below! Document your code as you write it. Run your changes against any existing tests and add new ones as needed to validate your changes; make sure you don’t accidentally break existing functionality! Several common commands can be accessed via the `make`:

        $ make lint

3. **Push commits to your forked repository:** Group changes into atomic git commits, then push them to your `origin` repository. There's no need to wait until all changes are final before pushing — it's always good to have a backup, in case something goes wrong in your local copy.

        $ git push origin my-descriptive-branch-name

4. **Open a new Pull Request in GitHub:** When you're ready to submit your changes to the main repo, navigate to your forked repository on GitHub. Switch to your working branch then click "New pull request"; alternatively, if you recently pushed, you may see a banner at the top of the repo with a "Compare & pull request" button, which you can click on to initiate the same process. Fill out the PR template completely and clearly, confirm that the code "diff" is as expected, then submit the PR. A number of processes will run automatically via GitHub Workflows (see `.github/workflows/`); we'll want to make sure everything passes before the PR gets merged.

5. **Respond to any code review feedback:** At this point, @Hironsan will review your work and either request additional changes/clarification or approve your work. There may be some necessary back-and-forth; please do your best to be responsive. If you haven’t gotten a response in a week or so, please politely nudge him in the same thread — thanks in advance for your patience!

## Styleguides

### Git Commit Messages

* Use the present tense ("Add feature" not "Added feature")
* Use the imperative mood ("Move cursor to..." not "Moves cursor to...")
* Limit the first line to 72 characters or less
* Reference issues and pull requests liberally after the first line
