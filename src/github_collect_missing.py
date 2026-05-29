"""
Append repos to under-populated class CSVs without using GitHub Search.
Run this after github_collector.py when some classes are below target.
"""

import sys
import time
import pandas as pd
from pathlib import Path

# Import shared API helpers from sibling module
sys.path.insert(0, str(Path(__file__).parent))
from github_collector import check_auth, fetch_repo, get_repo_signals, _checkpoint_path

# ── targets ────────────────────────────────────────────────────────────────────
TARGETS = {
    "lead_architect":      130,
    "template_boilerplate": 100,
    "low_value":           100,
}

# ── additional curated lists ───────────────────────────────────────────────────
# These complement github_collector.py's lists.
# Dedup is done at runtime against the existing CSVs, so overlaps are harmless.
ADDITIONAL = {
    "lead_architect": [
        # Cloud-native / Kubernetes ecosystem
        "kubernetes/helm", "kubernetes/minikube", "kubernetes/client-go",
        "kubernetes/dashboard", "kubernetes/kops", "kubernetes/kube-state-metrics",
        "kubernetes/ingress-nginx", "kubernetes/autoscaler",
        "argoproj/argo-cd", "argoproj/argo-workflows",
        "fluxcd/flux2", "fluxcd/flagger",
        "open-policy-agent/opa", "open-policy-agent/gatekeeper",
        "kubeflow/kubeflow", "rook/rook", "vitess/vitess",
        "cilium/hubble", "cilium/tetragon",
        # HashiCorp ecosystem
        "hashicorp/vault", "hashicorp/consul", "hashicorp/nomad",
        "hashicorp/packer", "hashicorp/boundary", "hashicorp/waypoint",
        "hashicorp/terraform-provider-aws", "hashicorp/terraform-provider-google",
        "hashicorp/terraform-provider-azurerm",
        "pulumi/pulumi", "pulumi/pulumi-aws",
        # Databases / storage
        "postgres/postgres", "mysql/mysql-server", "openssl/openssl",
        "redis/redis", "mongodb/mongo",
        "tikv/tikv", "pingcap/tidb",
        "ClickHouse/ClickHouse", "questdb/questdb",
        "yugabyte/yugabyte-db", "dgraph-io/dgraph",
        "timescale/timescaledb", "apache/cassandra",
        "apache/hbase", "apache/hive", "apache/druid",
        "apache/zookeeper", "apache/lucene",
        "trino/trino", "prestodb/presto",
        # Messaging / streaming
        "nats-io/nats-server", "rabbitmq/rabbitmq-server",
        "apache/pulsar", "apache/rocketmq", "apache/activemq",
        # ML / AI frameworks
        "keras-team/keras", "Lightning-AI/pytorch-lightning",
        "fastai/fastai", "onnx/onnx", "microsoft/onnxruntime",
        "triton-lang/triton", "jax-ml/jax", "google/flax",
        "langchain-ai/langchain", "microsoft/semantic-kernel",
        "openai/openai-python", "intel/openvino",
        "PaddlePaddle/Paddle", "apache/mxnet",
        # Observability
        "grafana/loki", "grafana/tempo", "grafana/mimir",
        "elastic/kibana", "elastic/logstash", "elastic/beats",
        "influxdata/telegraf", "influxdata/chronograf",
        # Runtimes / compilers / OS
        "apple/swift", "apple/swift-package-manager",
        "JetBrains/kotlin", "dotnet/aspnetcore", "dotnet/roslyn", "dotnet/sdk",
        "systemd/systemd", "qemu/qemu", "nginx/nginx", "haproxy/haproxy",
        # Cloud SDKs
        "aws/aws-cdk", "microsoft/azure-sdk-for-python",
        "googleapis/google-cloud-python", "alibaba/nacos",
        "alibaba/sentinel", "alibaba/seata", "apache/dubbo",
        # Frontend ecosystem
        "facebook/react-native", "facebook/jest", "facebook/docusaurus",
        "microsoft/playwright", "microsoft/TypeScript",
        "angular/angular", "angular/components",
        "spring-projects/spring-framework", "spring-projects/spring-boot",
        # Misc large OSS
        "protocolbuffers/protobuf", "google/flatbuffers",
        "google/googletest", "abseil/abseil-cpp", "abseil/abseil-py",
        "facebook/rocksdb", "facebook/folly", "facebook/proxygen",
        "mozilla/gecko-dev", "hibernate/hibernate-orm",
    ],

    "template_boilerplate": [
        # Python project templates
        "cookiecutter/cookiecutter-pylibrary",
        "pytest-dev/cookiecutter-pytest-plugin",
        "jacebrowning/template-python",
        "navdeep-G/samplemod",
        "pypa/sampleproject",
        "nickvdyck/cookiecutter-go",
        "nickvdyck/cookiecutter-csharp-standard",
        # Docker-based starters
        "tiangolo/uvicorn-gunicorn-fastapi-docker",
        "tiangolo/uwsgi-nginx-flask-docker",
        "tiangolo/meinheld-gunicorn-flask-docker",
        "tiangolo/uwsgi-nginx-docker",
        "nickjj/docker-django-example",
        "nickjj/docker-flask-example",
        "nickjj/docker-rails-example",
        "nickjj/docker-node-example",
        "nickjj/docker-go-example",
        "nickjj/docker-phoenix-example",
        # Frontend / JS starters
        "h5bp/html5-boilerplate",
        "h5bp/server-configs-nginx",
        "facebook/create-react-app",
        "vuejs/vue-cli",
        "angular/angular-cli",
        "electron/electron-quick-start",
        "electron/electron-api-demos",
        "vercel/commerce",
        "vercel/nextjs-subscription-payments",
        "vercel/ai-chatbot",
        "sveltejs/kit",
        "sveltejs/template",
        # GitHub / Codespaces official
        "github/codespaces-blank",
        "github/codespaces-react",
        "github/codespaces-flask",
        "github/codespaces-jupyter",
        "github/codespaces-django",
        # IDE / generator tooling
        "microsoft/vscode-generator-code",
        "yeoman/yo",
        # GitPod templates
        "gitpod-io/template-python-django",
        "gitpod-io/template-python-flask",
        "gitpod-io/template-node",
        "gitpod-io/template-rust",
        "gitpod-io/template-golang",
        "gitpod-io/template-java-spring",
        "gitpod-io/template-typescript",
        "gitpod-io/template-svelte",
        # Framework-specific demo / starter apps
        "laravel/laravel",
        "laravel/jetstream",
        "laravel/breeze",
        "laravel/sail",
        "thoughtbot/suspenders",
        "symfony/demo",
        "spring-projects/spring-petclinic",
        "samdark/yii2-starter-kit",
        "yiisoft/yii2-app-basic",
        # AI / ML project starters
        "microsoft/azure-search-openai-demo",
        "microsoft/chat-copilot",
        "openai/openai-quickstart-python",
        "anthropics/anthropic-quickstarts",
        "langchain-ai/langchain-template",
        "langchain-ai/langserve",
        # JetBrains / Mobile templates
        "JetBrains/kotlin-multiplatform-template",
        "JetBrains/compose-multiplatform-template",
        "JetBrains/kotlin-js-browser-template",
        "android/architecture-templates",
        "android/compose-samples",
        # Misc popular starters
        "nicholasjackson/terraform-hashicat-aws",
        "nicholasjackson/terraform-hashicat-azure",
        "hashicorp/learn-terraform-provision-eks-cluster",
        "hashicorp/learn-terraform-azure",
        "gruntwork-io/terragrunt-infrastructure-live-example",
        "trussworks/terraform-aws-template",
    ],

    "low_value": [
        # GitHub's own demo / test repos
        "octocat/hello-world", "octocat/Spoon-Knife",
        "octocat/git-consortium", "octocat/Hello-World",
        "github/testrepo", "github/github-services",
        "github/markup",
        # Personal dotfiles (single config files, no real engineering)
        "holman/dotfiles", "mathiasbynens/dotfiles",
        "paulirish/dotfiles", "ryanb/dotfiles",
        "anishathalye/dotfiles", "cowboy/dotfiles",
        "necolas/dotfiles", "alrra/dotfiles",
        "webpro/dotfiles", "gf3/dotfiles",
        "ptb/dotfiles", "driesvints/dotfiles",
        "joshukraine/dotfiles", "nicholasjackson/dotfiles",
        "superbrothers/dotfiles", "gpakosz/.tmux",
        "skwp/dotfiles", "thoughtbot/dotfiles",
        "jessfraz/dotfiles", "caarlos0/dotfiles",
        "atomantic/dotfiles", "Parth/dotfiles",
        "dikiaap/dotfiles", "nicktindall/dotfiles",
        "nicksp/dotfiles", "jacobwgillespie/dotfiles",
        "sapegin/dotfiles", "claytron/dotfiles",
        # Old / abandoned projects (early GitHub 2008-2012 era)
        "mojombo/bert", "mojombo/god", "mojombo/chronic",
        "defunkt/grub", "defunkt/ambition",
        "defunkt/cache-money", "defunkt/fixture-scenarios",
        "defunkt/facets", "defunkt/choices",
        "wycats/merb", "wycats/merb-core", "wycats/thor",
        "jashkenas/coffee-script",
        "rails/prototype-rails", "rails/arel", "rails/journey",
        "rails/strong_parameters",
        "thoughtbot/factory_girl", "thoughtbot/paperclip",
        "thoughtbot/shoulda", "thoughtbot/hoptoad_notifier",
        "rspec/rspec",
        # Demo / broken / workshop repos with no real development
        "nicholasjackson/broken-hashiapp",
        "nicholasjackson/http-echo",
        "nicholasjackson/terraform-null-test",
        "nicholasjackson/vault-init",
        "nicholasjackson/microservice-kubernetes-demo",
        "nicholasjackson/raspberry-pi",
        "nicholasjackson/k8s-alpha",
        "nicholasjackson/sentinel-workshop",
        "nicholasjackson/demo-nomad-gitops",
        "nicholasjackson/demo-consul-service-mesh",
        "nicholasjackson/demo-k8s-canary",
        "nicholasjackson/kubeadm-vagrant",
        "nicholasjackson/consul-k8s-l7-demo",
        "nicholasjackson/modern-application-workshop",
        "nicholasjackson/service-mesh-workshop",
        "nicholasjackson/kubecon-demo",
        "nicholasjackson/vault-plugin-demo",
        "nicholasjackson/nomad-canary",
        "nicholasjackson/fake-service",
        "nicholasjackson/demo",
        "nicholasjackson/test",
        "nicholasjackson/temp",
        # Deprecated archived official repos
        "pjhyett/github-services",
        "github/android", "github/ios",
        "github/gitignore-boilerplates",
        "github/gov-takedowns",
        "blog/Octopress",
        "imathis/octopress",
        "plusjade/jekyll-bootstrap",
        "barryclark/jekyll-now",
        "mmistakes/minimal-mistakes",
        # Minimal personal repos / first projects
        "defunkt/dotfiles", "mojombo/dotfiles",
        "ekmett/dotfiles", "wycats/dotfiles",
        "rtomayko/dotfiles", "pjhyett/dotfiles",
        "schacon/whygitisbetterthanx",
        "schacon/showoff",
    ],
}


# ── core logic ─────────────────────────────────────────────────────────────────

def load_existing(label: str) -> tuple[pd.DataFrame, set]:
    """Return (existing DataFrame, set of already-seen full_names)."""
    ckpt = _checkpoint_path(label)
    if ckpt.exists():
        df = pd.read_csv(ckpt)
        return df, set(df["full_name"].str.lower())
    return pd.DataFrame(), set()


def fill_class(label: str, repo_list: list, target: int) -> None:
    existing_df, seen = load_existing(label)
    current = len(existing_df)

    if current >= target:
        print(f"[{label}] already at {current}/{target} — skipping")
        return

    needed = target - current
    print(f"\n[{label}] have {current}, need {needed} more → target {target}")

    new_records = []
    for full_name in repo_list:
        if len(new_records) >= needed:
            break
        if full_name.lower() in seen:
            print(f"  skip (duplicate): {full_name}")
            continue
        seen.add(full_name.lower())

        print(f"  fetching: {full_name}")
        repo = fetch_repo(full_name)
        if repo is None:
            print(f"    → 404 / error")
            continue

        new_records.append(get_repo_signals(repo, label))
        time.sleep(1.5)

    if not new_records:
        print(f"  No new repos collected for {label}")
        return

    combined = (
        pd.concat([existing_df, pd.DataFrame(new_records)], ignore_index=True)
        if not existing_df.empty
        else pd.DataFrame(new_records)
    )
    combined.to_csv(_checkpoint_path(label), index=False)
    print(f"  [{label}] saved {len(combined)} total → data/raw/{label}.csv")


def rebuild_combined() -> None:
    """Merge all per-class CSVs into data/raw/repositories.csv."""
    frames = [
        pd.read_csv(f)
        for f in Path("data/raw").glob("*.csv")
        if f.name != "repositories.csv" and f.stat().st_size > 0
    ]
    if not frames:
        print("No class CSVs found — nothing to merge.")
        return
    combined = pd.concat(frames, ignore_index=True).drop_duplicates(subset="repo_id")
    combined.to_csv("data/raw/repositories.csv", index=False)
    print(f"\nRebuilt repositories.csv → {len(combined)} total repos")
    print(combined["label"].value_counts().to_string())


def main() -> None:
    print("Verifying GitHub authentication …")
    check_auth()

    for label, repo_list in ADDITIONAL.items():
        fill_class(label, repo_list, TARGETS[label])

    rebuild_combined()


if __name__ == "__main__":
    main()
