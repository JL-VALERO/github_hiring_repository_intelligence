"""GitHub API extraction — collects repository-level signals."""

import os
import re
import time
import requests
import pandas as pd
from pathlib import Path
from datetime import datetime, timezone
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

TOKEN = os.getenv("GITHUB_TOKEN", "").strip()
if not TOKEN:
    raise ValueError("GITHUB_TOKEN not found. Copy .env.example to .env and add your token.")

HEADERS = {
    "Authorization": f"token {TOKEN}",
    "Accept": "application/vnd.github+json",
    "X-GitHub-Api-Version": "2022-11-28",
    "User-Agent": "github-hiring-repository-intelligence/1.0",
}
BASE = "https://api.github.com"

# Curated lists — direct GET /repos/{owner}/{name}, no Search API, no secondary rate limits.
REPOS = {
    "lead_architect": [
        # Core OS / compilers / runtimes
        "torvalds/linux", "llvm/llvm-project", "golang/go",
        "rust-lang/rust", "JuliaLang/julia", "apple/swift",
        "dotnet/runtime", "dotnet/aspnetcore", "dotnet/roslyn",
        "mozilla/gecko-dev", "systemd/systemd", "qemu/qemu",
        "openssl/openssl", "postgres/postgres", "mysql/mysql-server",
        # Cloud-native / infra
        "kubernetes/kubernetes", "kubernetes/helm", "kubernetes/minikube",
        "kubernetes/client-go", "containerd/containerd",
        "moby/moby", "docker/compose", "docker/buildx",
        "istio/istio", "envoyproxy/envoy", "cilium/cilium",
        "etcd-io/etcd", "argoproj/argo-cd", "argoproj/argo-workflows",
        "fluxcd/flux2", "open-policy-agent/opa", "kubeflow/kubeflow",
        "rook/rook", "vitess/vitess", "hashicorp/terraform",
        "hashicorp/vault", "hashicorp/consul", "hashicorp/nomad",
        "hashicorp/packer", "pulumi/pulumi",
        "hashicorp/terraform-provider-aws",
        # Observability / data stores
        "prometheus/prometheus", "grafana/grafana", "grafana/loki",
        "grafana/tempo", "grafana/mimir", "elastic/elasticsearch",
        "elastic/kibana", "elastic/logstash", "influxdata/influxdb",
        "timescale/timescaledb", "questdb/questdb",
        "cockroachdb/cockroach", "tikv/tikv", "pingcap/tidb",
        "ClickHouse/ClickHouse", "dgraph-io/dgraph",
        "mongodb/mongo", "redis/redis", "minio/minio",
        # Streaming / messaging
        "apache/kafka", "apache/pulsar", "nats-io/nats-server",
        "rabbitmq/rabbitmq-server", "apache/rocketmq",
        # Big data / ML infra
        "apache/spark", "apache/flink", "apache/beam",
        "apache/airflow", "apache/arrow", "apache/hadoop",
        "apache/cassandra", "apache/druid", "apache/zookeeper",
        "apache/hbase", "apache/hive", "trino/trino",
        "dask/dask", "ray-project/ray", "prefecthq/prefect",
        "dagster-io/dagster",
        # ML / AI frameworks
        "tensorflow/tensorflow", "pytorch/pytorch", "keras-team/keras",
        "huggingface/transformers", "fastai/fastai",
        "Lightning-AI/pytorch-lightning", "onnx/onnx",
        "microsoft/onnxruntime", "triton-lang/triton", "jax-ml/jax",
        "google/flax", "intel/openvino", "langchain-ai/langchain",
        "microsoft/semantic-kernel", "openai/openai-python",
        # Web / frontend frameworks
        "facebook/react", "facebook/react-native",
        "vuejs/vue", "angular/angular", "sveltejs/svelte",
        "nuxt/nuxt", "vitejs/vite", "nestjs/nest",
        "denoland/deno", "microsoft/TypeScript",
        # Backend frameworks / tooling
        "django/django", "ansible/ansible",
        "spring-projects/spring-framework", "spring-projects/spring-boot",
        "JetBrains/kotlin", "hibernate/hibernate-orm",
        "facebook/jest", "facebook/docusaurus",
        "microsoft/playwright", "microsoft/vscode",
        "microsoft/terminal", "microsoft/azure-sdk-for-python",
        "home-assistant/core", "grpc/grpc",
        "protocolbuffers/protobuf", "google/flatbuffers",
        "google/googletest", "abseil/abseil-py", "abseil/abseil-cpp",
        "facebook/rocksdb", "facebook/folly",
        "aws/aws-cdk", "alibaba/nacos", "alibaba/sentinel",
        "apache/dubbo", "open-telemetry/opentelemetry-python",
        "yugabyte/yugabyte-db", "apache/lucene",
    ],
    "senior": [
        "scikit-learn/scikit-learn", "pandas-dev/pandas", "numpy/numpy",
        "psf/requests", "psf/black", "tiangolo/fastapi",
        "encode/django-rest-framework", "sqlalchemy/sqlalchemy",
        "aio-libs/aiohttp", "tqdm/tqdm", "pytest-dev/pytest",
        "sphinx-doc/sphinx", "mkdocs/mkdocs", "pypa/pip",
        "matplotlib/matplotlib", "bokeh/bokeh", "streamlit/streamlit",
        "explosion/spaCy", "nltk/nltk", "networkx/networkx",
        "pydantic/pydantic", "encode/httpx", "pallets/click",
        "Textualize/rich", "scrapy/scrapy", "paramiko/paramiko",
        "marshmallow-code/marshmallow", "arrow-py/arrow",
        "pallets/flask", "celery/celery", "twisted/twisted",
        "python-poetry/poetry", "PyCQA/isort", "PyCQA/flake8",
        "pre-commit/pre-commit", "tiangolo/typer",
        "encode/starlette", "encode/uvicorn",
        "plotly/plotly.py", "altair-viz/altair",
        "pypa/setuptools", "psf/mypy", "python/cpython",
        "boto/boto3", "redis/redis-py", "sqlalchemy/alembic",
        "joke2k/faker", "aws/aws-cli", "dateutil/dateutil",
        "yaml/pyyaml", "pypa/virtualenv", "encode/broadcaster",
    ],
    "template_boilerplate": [
        # Python cookiecutters & project templates
        "cookiecutter/cookiecutter",
        "audreyfeldroy/cookiecutter-pypackage",
        "cjolowicz/cookiecutter-hypermodern-python",
        "pydanny/cookiecutter-django",
        "drivendata/cookiecutter-data-science",
        "cookiecutter/cookiecutter-pylibrary",
        "cookiecutter/cookiecutter-flask",
        "pytest-dev/cookiecutter-pytest-plugin",
        "wemake-services/wemake-django-template",
        "rochacbruno/python-project-template",
        "jacebrowning/template-python",
        "navdeep-G/samplemod",
        "pypa/sampleproject",
        "MichaelCurrin/py-project-template",
        "microsoft/python-package-template",
        # FastAPI / Flask / Django templates
        "tiangolo/full-stack-fastapi-template",
        "fastapi/full-stack-fastapi-template",
        "tiangolo/uvicorn-gunicorn-fastapi-docker",
        "tiangolo/uwsgi-nginx-flask-docker",
        "tiangolo/meinheld-gunicorn-flask-docker",
        "heroku/python-getting-started",
        "nickjj/docker-django-example",
        "nickjj/docker-flask-example",
        "nickjj/docker-rails-example",
        "nickjj/docker-node-example",
        "nickjj/docker-go-example",
        # Frontend / JS starters
        "h5bp/html5-boilerplate",
        "facebook/create-react-app",
        "vuejs/create-vue",
        "vuejs/vue-cli",
        "angular/angular-cli",
        "electron/electron-quick-start",
        "electron/electron-api-demos",
        "vercel/commerce",
        "vercel/nextjs-subscription-payments",
        "sveltejs/template",
        "sveltejs/kit",
        # GitHub / Codespaces official templates
        "github/gitignore",
        "github/codespaces-blank",
        "github/codespaces-react",
        "github/codespaces-flask",
        "github/codespaces-jupyter",
        "github/codespaces-django",
        "github/codespaces-rails",
        # IDE / tool generators
        "microsoft/vscode-extension-samples",
        "microsoft/vscode-generator-code",
        "yeoman/yo",
        "nickvdyck/cookiecutter-csharp-standard",
        "nickvdyck/cookiecutter-go",
        # GitPod templates
        "gitpod-io/template-python-django",
        "gitpod-io/template-python-flask",
        "gitpod-io/template-node",
        "gitpod-io/template-rust",
        "gitpod-io/template-golang",
        "gitpod-io/template-java-spring",
        "gitpod-io/template-typescript",
        # Framework-specific demo / starter apps
        "spring-projects/spring-petclinic",
        "laravel/laravel",
        "laravel/jetstream",
        "laravel/breeze",
        "thoughtbot/suspenders",
        "symfony/demo",
        "nicholasjackson/terraform-hashicat-aws",
        # ML / AI project templates
        "microsoft/aml-template",
        "microsoft/azure-search-openai-demo",
        "microsoft/chat-copilot",
        "openai/openai-cookbook",
        "microsoft/vscode-ai-toolkit",
        # JetBrains / mobile templates
        "JetBrains/kotlin-multiplatform-template",
        "JetBrains/compose-multiplatform-template",
        "JetBrains/kotlin-js-browser-template",
        "google/android-studio-poet",
        "android/architecture-templates",
    ],
    "junior": [
        "geekcomputers/Python",
        "CoreyMSchafer/code_snippets",
        "Pierian-Data/Complete-Python-3-Bootcamp",
        "avinashkranjan/Amazing-Python-Scripts",
        "Avik-Jain/100-Days-Of-ML-Code",
        "lmoroney/dlaicourse",
        "graykode/nlp-tutorial",
        "bregman-arie/devops-exercises",
        "donnemartin/interactive-coding-challenges",
        "jakevdp/PythonDataScienceHandbook",
        "wesm/pydata-book",
        "fastai/fastbook",
        "ageron/handson-ml2",
        "Asabeneh/30-Days-Of-Python",
        "Asabeneh/30-Days-Of-JavaScript",
        "bradtraversy/50projects50days",
        "bradtraversy/python-cheat-sheet",
        "florinpop17/app-ideas",
        "gothinkster/realworld",
        "karan/Projects-Solutions",
        "realpython/materials",
        "realpython/python-basics-exercises",
        "joeyajames/Python",
        "zhiwehu/Python-programming-exercises",
        "jerry-git/learn-python3",
        "Akuli/python-tutorial",
        "trekhleb/javascript-algorithms",
        "firstcontributions/first-contributions",
        "vinta/awesome-python",
        "TheAlgorithms/Python",
    ],
    "intern": [
        "bradtraversy/50projects50days",
        "florinpop17/100Days100Projects",
        "CodeWithHarry/Python-Mini-Projects",
        "Vishesh-Pandey/python-beginner-projects",
        "tiagoft/audio_for_ml",
        "mdipierro/nlib",
        "zhiwehu/Python-programming-exercises",
        "nicholasess/awesome-beginners",
        "ProgrammingHero1/100-plus-python-coding-problems-with-solution",
        "naomifridman/Introduction_to_neural_networks",
        "atapour/action-recognition",
        "nicholasjackson/demo-nomad-gitops",
        "ynagatomo/ARPhysics",
        "aceking007/100ProjectsOfCode",
        "nicholasjackson/demo-consul-service-mesh",
        "nicholasjackson/demo-k8s-canary",
        "nicholasjackson/kubeadm-vagrant",
        "ossamafariss/Learn-Go-With-Tests",
        "nicholasjackson/fake-service",
        "nicholasjackson/sentinel-workshop",
    ],
    "low_value": [
        # GitHub's own demo / test repos (no real code)
        "octocat/hello-world",
        "octocat/Spoon-Knife",
        "octocat/linguist",
        "octocat/git-consortium",
        "octocat/Hello-World",
        "github/testrepo",
        "github-classroom/autograding-example-python",
        # Personal dotfiles (single-purpose, minimal engineering value)
        "defunkt/dotfiles",
        "mojombo/dotfiles",
        "wycats/dotfiles",
        "ekmett/dotfiles",
        "holman/dotfiles",
        "mathiasbynens/dotfiles",
        "paulirish/dotfiles",
        "ryanb/dotfiles",
        "anishathalye/dotfiles",
        "cowboy/dotfiles",
        "necolas/dotfiles",
        "alrra/dotfiles",
        "webpro/dotfiles",
        "gf3/dotfiles",
        "ptb/dotfiles",
        # Old / abandoned projects from early GitHub
        "mojombo/bert",
        "mojombo/god",
        "mojombo/chronic",
        "defunkt/grub",
        "defunkt/ambition",
        "defunkt/cache-money",
        "defunkt/fixture-scenarios",
        "defunkt/facets",
        "wycats/merb",
        "wycats/merb-core",
        "wycats/thor",
        # Demo / broken / test repos with no active development
        "nicholasjackson/broken-hashiapp",
        "nicholasjackson/http-echo",
        "nicholasjackson/terraform-null-test",
        "nicholasjackson/vault-init",
        "nicholasjackson/old-project",
        "nicholasjackson/microservice-kubernetes-demo",
        "nicholasjackson/raspberry-pi",
        "nicholasjackson/k8s-alpha",
        "nicholasjackson/sentinel-workshop",
        "nicholasjackson/demo-nomad-gitops",
        "nicholasjackson/demo-consul-service-mesh",
        "nicholasjackson/demo-k8s-canary",
        "nicholasjackson/kubeadm-vagrant",
        "nicholasjackson/test",
        "nicholasjackson/temp",
        # Deprecated / archived official repos
        "github/github-services",
        "github/markup",
        "jashkenas/coffee-script",
        "rails/prototype-rails",
        "rails/arel",
        "rails/journey",
        "rails/strong_parameters",
        "rspec/rspec",
        "rspec/rspec-mocks",
        "thoughtbot/factory_girl",
        "thoughtbot/paperclip",
        "thoughtbot/shoulda",
        "thoughtbot/hoptoad_notifier",
        # Empty / placeholder / skeleton repos
        "nickvdyck/empty-repo",
        "nicholasjackson/kubecon-demo",
        "nicholasjackson/service-mesh-workshop",
        "nicholasjackson/modern-application-workshop",
        "nicholasjackson/consul-k8s-l7-demo",
        "nicholasjackson/consul-release-controller",
        "nicholasjackson/fake-service-old",
        # Simple "my first X" personal repos known to be trivial
        "nicholasess/awesome-beginners",
        "nicholasjackson/demo",
        "nicholasjackson/fake-service",
        "nicholasjackson/vault-plugin-demo",
        "nicholasjackson/nomad-canary",
    ],
}

_CI_FILES     = {".travis.yml", "jenkinsfile", ".gitlab-ci.yml", ".circleci"}
_TEST_FOLDERS = {"tests", "test", "__tests__", "spec", "specs"}
_REQ_FILES    = {
    "requirements.txt", "pyproject.toml", "setup.cfg",
    "package.json", "pom.xml", "go.mod", "cargo.toml",
}


def _get(url: str, params: dict | None = None) -> requests.Response | None:
    """GET with automatic rate-limit and Retry-After handling."""
    for attempt in range(3):
        try:
            resp = requests.get(url, headers=HEADERS, params=params, timeout=10)
        except requests.RequestException:
            time.sleep(2 ** attempt)
            continue

        if resp.status_code == 200:
            return resp

        if resp.status_code in (429, 403):
            # Print the exact GitHub error message for diagnosis
            try:
                msg = resp.json().get("message", "no message")
            except Exception:
                msg = resp.text[:200]
            print(f"  [403/429] GitHub says: {msg}")

            retry_after = resp.headers.get("Retry-After")
            remaining   = int(resp.headers.get("X-RateLimit-Remaining", 1))

            if retry_after:
                # GitHub explicitly told us how long to wait
                wait = int(retry_after) + 1
                print(f"  [rate limit] Retry-After={retry_after}s → sleeping {wait}s")
            elif remaining == 0:
                # Primary rate limit exhausted — sleep until the reset timestamp
                reset = int(resp.headers.get("X-RateLimit-Reset", time.time() + 60))
                wait  = max(reset - time.time(), 1) + 2
                print(f"  [primary rate limit] sleeping {wait:.0f}s until reset …")
            else:
                # Secondary (abuse) rate limit — remaining quota is fine but GitHub
                # is throttling the request pattern. Sleep 60s and retry.
                wait = 60
                print(f"  [secondary rate limit] sleeping {wait}s (remaining={remaining}) …")

            time.sleep(wait)

        elif resp.status_code in (404, 409, 422):
            return None
        else:
            time.sleep(2 ** attempt)
    return None


def fetch_repo(full_name: str) -> dict | None:
    """Fetch repo metadata directly — uses core rate limit (5000/hr), not search."""
    owner, name = full_name.split("/", 1)
    resp = _get(f"{BASE}/repos/{owner}/{name}")
    return resp.json() if resp else None


def _extract_root_signals(owner: str, name: str) -> dict:
    defaults = {
        "has_readme": False, "readme_size_bytes": 0,
        "has_docker": False, "has_requirements": False,
        "has_tests_folder": False, "has_ci_cd": False,
        "root_file_count": 0,
    }
    resp = _get(f"{BASE}/repos/{owner}/{name}/contents/")
    if resp is None:
        return defaults

    items = resp.json()
    if not isinstance(items, list):
        return defaults

    signals = dict(defaults)
    signals["root_file_count"] = len(items)
    for item in items:
        n = item["name"].lower()
        t = item.get("type", "")
        if n.startswith("readme"):
            signals["has_readme"] = True
            signals["readme_size_bytes"] = item.get("size", 0)
        if n == "dockerfile":
            signals["has_docker"] = True
        if n in _REQ_FILES:
            signals["has_requirements"] = True
        if n in _TEST_FOLDERS and t == "dir":
            signals["has_tests_folder"] = True
        if n == ".github" and t == "dir":
            signals["has_ci_cd"] = True
        if n in _CI_FILES:
            signals["has_ci_cd"] = True
    return signals


def _get_contributor_count(owner: str, name: str) -> int:
    resp = _get(
        f"{BASE}/repos/{owner}/{name}/contributors",
        params={"per_page": 1, "anon": "false"},
    )
    if resp is None:
        return 0
    link = resp.headers.get("Link", "")
    m = re.search(r'page=(\d+)>; rel="last"', link)
    if m:
        return int(m.group(1))
    data = resp.json()
    return len(data) if isinstance(data, list) else 0


def _days_since(iso_str: str) -> int | None:
    if not iso_str:
        return None
    now = datetime.now(timezone.utc)
    dt = datetime.fromisoformat(iso_str.replace("Z", "+00:00"))
    return (now - dt).days


def get_repo_signals(repo: dict, label: str) -> dict:
    owner = repo["owner"]["login"]
    name  = repo["name"]

    root = _extract_root_signals(owner, name)
    time.sleep(2)
    contributors = _get_contributor_count(owner, name)
    time.sleep(2)

    return {
        "label":                label,
        "repo_id":              repo["id"],
        "full_name":            repo["full_name"],
        "description":          (repo.get("description") or "")[:300],
        "language":             repo.get("language") or "",
        "topics":               ",".join(repo.get("topics", [])),
        "stars":                repo["stargazers_count"],
        "forks":                repo["forks_count"],
        "open_issues":          repo["open_issues_count"],
        "contributor_count":    contributors,
        "size_kb":              repo["size"],
        "is_fork":              repo["fork"],
        "is_archived":          repo.get("archived", False),
        "last_commit_days_ago": _days_since(repo.get("pushed_at")),
        "repo_age_days":        _days_since(repo.get("created_at")),
        **root,
    }


def _checkpoint_path(label: str) -> Path:
    return Path("data/raw") / f"{label}.csv"


def _save_checkpoint(label: str, records: list) -> None:
    pd.DataFrame(records).to_csv(_checkpoint_path(label), index=False)
    print(f"  [checkpoint] {len(records)} repos → data/raw/{label}.csv")


def collect_class(label: str, repo_list: list) -> list:
    records = []
    for full_name in repo_list:
        print(f"  [{label}] {full_name}")
        repo = fetch_repo(full_name)
        if repo is None:
            print(f"    skip (404 or error)")
            continue
        records.append(get_repo_signals(repo, label))
        time.sleep(2)
    return records


def check_auth() -> None:
    resp = _get(f"{BASE}/rate_limit")
    if resp is None:
        raise RuntimeError("Auth check failed — token may be invalid or network is down.")
    data = resp.json()
    core   = data["resources"]["core"]
    search = data["resources"]["search"]
    print(f"  Auth OK  |  core: {core['remaining']}/{core['limit']}  |  search: {search['remaining']}/{search['limit']}")
    if core["limit"] == 60:
        raise RuntimeError(
            "GitHub is treating requests as UNAUTHENTICATED (limit=60).\n"
            "Check that your .env contains GITHUB_TOKEN=<token> with no spaces."
        )


def main():
    print("Verifying GitHub authentication …")
    check_auth()

    all_records = []
    for label, repo_list in REPOS.items():
        ckpt = _checkpoint_path(label)
        if ckpt.exists():
            existing = pd.read_csv(ckpt).to_dict("records")
            print(f"\n[skip] {label} — loaded {len(existing)} repos from checkpoint")
            all_records.extend(existing)
            continue

        print(f"\n{'='*50}\nClass: {label}  ({len(repo_list)} repos in list)\n{'='*50}")
        records = collect_class(label, repo_list)
        print(f"  Collected: {len(records)}")
        _save_checkpoint(label, records)
        all_records.extend(records)

    df = pd.DataFrame(all_records)
    out = "data/raw/repositories.csv"
    df.to_csv(out, index=False)
    print(f"\nTotal: {len(df)} repos → {out}")
    print(df["label"].value_counts().to_string())


if __name__ == "__main__":
    main()
