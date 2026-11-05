#!/usr/bin/env bash
# Run once on the server. Uses sudo to install k8s/rbac.yaml, then writes a
# restricted kubeconfig to ~/.kube/config so run.sh works without sudo.
#
# Refuses to overwrite an existing ~/.kube/config. The token is never printed.
#
# To revoke: rm ~/.kube/config, then with the admin kubeconfig delete the
# gpu-bench-runner-token Secret (or the objects in k8s/rbac.yaml).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BIN=/var/lib/rancher/rke2/bin/kubectl
ADMIN=(sudo "$BIN" --kubeconfig /etc/rancher/rke2/rke2.yaml)
OUT="$HOME/.kube/config"
SECRET=gpu-bench-runner-token
NS=gpu-bench

die() { echo "error: $*" >&2; exit 1; }

[[ ! -e "$OUT" ]] || die "$OUT already exists; not overwriting it"
sudo -v
"${ADMIN[@]}" apply -f "$ROOT/k8s/rbac.yaml"

# Kubernetes fills in the token shortly after the Secret is created.
for _ in $(seq 1 30); do
  token_b64="$("${ADMIN[@]}" get secret "$SECRET" -n "$NS" -o 'jsonpath={.data.token}')"
  [[ -z "$token_b64" ]] || break
  sleep 1
done
[[ -n "$token_b64" ]] || die "token was not issued for $SECRET"
ca_b64="$("${ADMIN[@]}" get secret "$SECRET" -n "$NS" -o 'jsonpath={.data.ca\.crt}')"

umask 077
mkdir -p "$(dirname "$OUT")"
cat >"$OUT" <<EOF
apiVersion: v1
kind: Config
clusters:
  - name: ampere-fractioning
    cluster:
      server: https://127.0.0.1:6443
      certificate-authority-data: $ca_b64
users:
  - name: gpu-bench-runner
    user:
      token: $(base64 -d <<<"$token_b64")
contexts:
  - name: gpu-bench
    context: {cluster: ampere-fractioning, user: gpu-bench-runner, namespace: $NS}
current-context: gpu-bench
EOF
chmod 600 "$OUT"

# Subresources need --subresource: `can-i create pods/exec` means a Pod named exec.
check() { # expected verb resource [flags...]
  local expected="$1"; shift
  local actual
  actual="$("$BIN" auth can-i "$@" 2>/dev/null || true)"
  printf '%-4s (expected %-3s) %s\n' "$actual" "$expected" "$*"
  [[ "$actual" == "$expected" ]] || FAILED=1
}

FAILED=0
check yes create pods -n "$NS"
check yes get pods --subresource=log -n "$NS"
check yes create configmaps -n "$NS"
check yes get gpufractioningconfigs.gpu-fractioning.kai.scheduler
check no create pods --subresource=exec -n "$NS"
check no create pods -n kube-system
check no delete gpufractioningconfigs.gpu-fractioning.kai.scheduler
check no get secrets -n "$NS"

[[ "$FAILED" == 0 ]] || die "permissions are not as expected; review $OUT"
echo "ok: restricted access installed at $OUT"
