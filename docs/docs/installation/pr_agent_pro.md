
## Getting Started with Qodo Merge Pro

Qodo Merge Pro is a versatile application compatible with GitHub, GitLab, and BitBucket, hosted by CodiumAI.
See [here](https://qodo-merge-docs.qodo.ai/overview/pr_agent_pro/) for more details about the benefits of using Qodo Merge Pro.

Interested parties can subscribe to Qodo Merge Pro through the following [link](https://www.codium.ai/pricing/).
After subscribing, you are granted the ability to easily install the application across any of your repositories.

![Qodo Merge Pro](https://codium.ai/images/pr_agent/pr_agent_pro_install.png){width=468}

Each user who wants to use Qodo Merge pro needs to buy a seat.
Initially, CodiumAI offers a two-week trial period at no cost, after which continued access requires each user to secure a personal seat.
Once a user acquires a seat, they gain the flexibility to use Qodo Merge Pro across any repository where it was enabled.

Users without a purchased seat who interact with a repository featuring Qodo Merge Pro are entitled to receive up to five complimentary feedbacks.
Beyond this limit, Qodo Merge Pro will cease to respond to their inquiries unless a seat is purchased.

## Install Qodo Merge Pro for GitHub Enterprise Server

To use Qodo Merge Pro application on your private GitHub Enterprise Server, you will need to contact us for starting an [Enterprise](https://www.codium.ai/pricing/) trial.


## Install Qodo Merge Pro for GitLab (Teams & Enterprise)

Since GitLab platform does not support apps, installing Qodo Merge Pro for GitLab is a bit more involved, and requires the following steps:

### Step 1

Acquire a personal, project or group level access token. Enable the “api” scope in order to allow Qodo Merge to read pull requests, comment and respond to requests.

<figure markdown="1">
![Step 1](https://www.codium.ai/images/pr_agent/gitlab_pro_pat.png){width=750}
</figure>

Store the token in a safe place, you won’t be able to access it again after it was generated.

### Step 2

Generate a shared secret and link it to the access token. Browse to [https://register.gitlab.pr-agent.codium.ai](https://register.gitlab.pr-agent.codium.ai).
Fill in your generated GitLab token and your company or personal name in the appropriate fields and click "Submit".

You should see "Success!" displayed above the Submit button, and a shared secret will be generated. Store it in a safe place, you won’t be able to access it again after it was generated.

### Step 3

Install a webhook for your repository or groups, by clicking “webhooks” on the settings menu. Click the “Add new webhook” button.

<figure markdown="1">
![Step 3.1](https://www.codium.ai/images/pr_agent/gitlab_pro_add_webhook.png)
</figure>

In the webhook definition form, fill in the following fields:
URL: https://pro.gitlab.pr-agent.codium.ai/webhook

Secret token: Your CodiumAI key
Trigger: Check the ‘comments’ and ‘merge request events’ boxes.
Enable SSL verification: Check the box.

<figure markdown="1">
![Step 3.2](https://www.codium.ai/images/pr_agent/gitlab_pro_webhooks.png){width=750}
</figure>

### Step 4

You’re all set!

Open a new merge request or add a MR comment with one of Qodo Merge’s commands such as /review, /describe or /improve.
