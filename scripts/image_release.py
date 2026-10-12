"""What an image a deployment runs is, and whether the registry has a newer one.

A deployment that runs published images -- the server image, a tool library's
prebuilt bundle -- follows a TAG (`:latest`, `:deploy`), not a branch. Whether
it is behind is whether the digest the registry now serves under that tag
differs from the digest of the image on this machine. What changed in between
comes from the image's own labels (`org.opencontainers.image.revision` and
`.source`, stamped when it was built) and the GitHub repository they name.

Standard library only, like every host script: it runs where nothing else is
installed, and asks the registry and GitHub anonymously, as any client may for
public images and repositories.
"""

import json
import re
import subprocess
import urllib.error
import urllib.parse
import urllib.request

_MANIFEST_TYPES = ", ".join((
    "application/vnd.oci.image.index.v1+json",
    "application/vnd.docker.distribution.manifest.list.v2+json",
    "application/vnd.oci.image.manifest.v1+json",
    "application/vnd.docker.distribution.manifest.v2+json",
))
_HEADERS = {"User-Agent": "visor-update-agent/1.0"}
_GITHUB = re.compile(r"github\.com[:/]([^/]+)/([^/]+?)(?:\.git)?/?$")


def split_reference(image: str):
    """`ghcr.io/owner/name:tag` -> ("ghcr.io", "owner/name", "tag").

    A reference with no registry host is Docker Hub's, and one with no tag is
    `latest`, as docker itself reads them.
    """
    name, tag = image, "latest"
    if "@" in name:
        name = name.split("@", 1)[0]
    last = name.rsplit("/", 1)[-1]
    if ":" in last:
        name, tag = name.rsplit(":", 1)
    first = name.split("/", 1)[0]
    if "/" in name and ("." in first or ":" in first or first == "localhost"):
        registry, repository = name.split("/", 1)
    else:
        registry, repository = "registry-1.docker.io", name if "/" in name else f"library/{name}"
    return registry, repository, tag


def _bearer(challenge: str, repository: str) -> str:
    """An anonymous pull token from the realm a 401 named, or ""."""
    fields = dict(re.findall(r'(\w+)="([^"]*)"', challenge or ""))
    realm = fields.get("realm")
    if not realm:
        return ""
    query = {"scope": fields.get("scope") or f"repository:{repository}:pull"}
    if fields.get("service"):
        query["service"] = fields["service"]
    try:
        with urllib.request.urlopen(urllib.request.Request(
                realm + "?" + urllib.parse.urlencode(query), headers=_HEADERS), timeout=20) as response:
            payload = json.loads(response.read().decode("utf-8") or "{}")
    except (OSError, ValueError):
        return ""
    return payload.get("token") or payload.get("access_token") or ""


def remote_digest(image: str, opener=urllib.request.urlopen) -> str:
    """The digest the registry serves under this image's tag, or "" when it
    cannot be asked (offline, private, unknown tag) -- never an exception."""
    registry, repository, tag = split_reference(image)
    url = f"https://{registry}/v2/{repository}/manifests/{tag}"
    headers = {**_HEADERS, "Accept": _MANIFEST_TYPES}
    for attempt in range(2):
        try:
            with opener(urllib.request.Request(url, headers=headers, method="HEAD"), timeout=20) as response:
                return response.headers.get("Docker-Content-Digest", "")
        except urllib.error.HTTPError as exc:
            if exc.code != 401 or attempt:
                return ""
            token = _bearer(exc.headers.get("WWW-Authenticate", ""), repository)
            if not token:
                return ""
            headers["Authorization"] = f"Bearer {token}"
        except OSError:
            return ""
    return ""


def local_image(image: str, run=subprocess.run) -> dict:
    """`{"present", "digest", "labels"}` for the image on this machine.

    `digest` is the registry digest it was pulled under, matched to this
    image's repository; an image built here and never pushed has none, and is
    then reported as not followable rather than as always behind.
    """
    try:
        done = run(["docker", "image", "inspect", image, "--format",
                    "{{json .RepoDigests}}\t{{json .Config.Labels}}"],
                   capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.SubprocessError):
        return {"present": False, "digest": "", "labels": {}}
    if done.returncode != 0:
        return {"present": False, "digest": "", "labels": {}}
    digests_json, _, labels_json = done.stdout.strip().partition("\t")
    try:
        digests = json.loads(digests_json) or []
        labels = json.loads(labels_json) or {}
    except ValueError:
        digests, labels = [], {}
    registry, repository, _tag = split_reference(image)
    wanted = {f"{registry}/{repository}", repository}
    digest = next((entry.split("@", 1)[1] for entry in digests
                   if "@" in entry and entry.split("@", 1)[0] in wanted), "")
    return {"present": True, "digest": digest, "labels": labels}


def github_repository(source: str):
    """`(owner, name)` of a github.com source URL, or None."""
    match = _GITHUB.search(source or "")
    return (match.group(1), match.group(2)) if match else None


def _github(path: str, opener=urllib.request.urlopen):
    try:
        with opener(urllib.request.Request(f"https://api.github.com/{path}", headers={
                **_HEADERS, "Accept": "application/vnd.github+json"}), timeout=20) as response:
            return json.loads(response.read().decode("utf-8") or "null")
    except (OSError, ValueError):
        return None


def latest_release(source: str, prefix: str, opener=urllib.request.urlopen) -> dict:
    """The newest published release named `<prefix>...` and the commit it was
    cut from, or {} -- the data and tool series share one repository."""
    repository = github_repository(source)
    if not repository:
        return {}
    releases = _github(f"repos/{repository[0]}/{repository[1]}/releases?per_page=30", opener) or []
    for release in releases:
        tag = release.get("tag_name") or ""
        if tag.startswith(prefix) and not release.get("draft"):
            ref = _github(f"repos/{repository[0]}/{repository[1]}/commits/{urllib.parse.quote(tag)}", opener) or {}
            return {"tag": tag, "revision": ref.get("sha", ""), "at": release.get("published_at")}
    return {}


def compare(source: str, base: str, head: str, opener=urllib.request.urlopen) -> dict:
    """Commits and changed paths from `base` to `head`, or {}."""
    repository = github_repository(source)
    if not (repository and base and head and base != head):
        return {}
    payload = _github(f"repos/{repository[0]}/{repository[1]}/compare/{base}...{head}", opener) or {}
    commits = []
    for commit in payload.get("commits") or []:
        detail = commit.get("commit") or {}
        commits.append({
            "sha": (commit.get("sha") or "")[:9],
            "subject": (detail.get("message") or "").splitlines()[0] if detail.get("message") else "",
            "author": (detail.get("author") or {}).get("name", ""),
            "at": None,
        })
    paths = [entry.get("filename") for entry in payload.get("files") or [] if entry.get("filename")]
    return {"commits": list(reversed(commits))[:30], "paths": paths, "behind": payload.get("ahead_by", len(commits))}
