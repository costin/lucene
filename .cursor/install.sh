#!/usr/bin/env bash
# Idempotent repository bootstrap for Apache Lucene Cloud Agents.
# Warms the Gradle wrapper distribution and dependency caches and compiles the
# core module so that subsequent agent builds/tests are fast. Safe to re-run.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${repo_root}"

# Prefer the JDK 25 provided by the base image, but tolerate it already being
# on PATH / JAVA_HOME when re-run in other contexts.
if [ -x /opt/jdk25/bin/java ]; then
  export JAVA_HOME=/opt/jdk25
  export PATH="${JAVA_HOME}/bin:${PATH}"
fi

echo "Using Java:"
java -version

# Download the pinned Gradle distribution and print the toolchain.
./gradlew --version

# Compile the core module. This resolves the bulk of the build dependencies and
# validates that the toolchain works. Gradle keeps this up-to-date on re-runs.
./gradlew :lucene:core:assemble
