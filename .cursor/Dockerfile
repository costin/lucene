# Cloud Agent base image for Apache Lucene.
# Lucene's main branch requires JDK 25 (see gradle/libs.versions.toml: minJava).
# System packages and the JDK are stable dependencies and belong here; project
# dependency resolution / compilation is handled by the install phase.
FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive

# git + curl are required in fully custom Cloud Agent images.
# perl and python3 are required by some Gradle checks (e.g. ./gradlew check).
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates \
        curl \
        git \
        perl \
        python3 \
        unzip \
        zip \
    && rm -rf /var/lib/apt/lists/*

# Install Eclipse Temurin JDK 25 (x86_64), pinned with checksum verification.
ENV JAVA_HOME=/opt/jdk25
ENV JDK_TARBALL=OpenJDK25U-jdk_x64_linux_hotspot_25.0.4.1_1.tar.gz
ENV JDK_URL=https://github.com/adoptium/temurin25-binaries/releases/download/jdk-25.0.4.1%2B1/OpenJDK25U-jdk_x64_linux_hotspot_25.0.4.1_1.tar.gz
ENV JDK_SHA256=dbb698396d478e7fa2b1e50f4103324b2a99b90569ee27c33f2261f9215cf41e
RUN mkdir -p "${JAVA_HOME}" \
    && curl -fsSL -o "/tmp/${JDK_TARBALL}" "${JDK_URL}" \
    && echo "${JDK_SHA256}  /tmp/${JDK_TARBALL}" | sha256sum -c - \
    && tar -xzf "/tmp/${JDK_TARBALL}" -C "${JAVA_HOME}" --strip-components=1 \
    && rm "/tmp/${JDK_TARBALL}"
ENV PATH="${JAVA_HOME}/bin:${PATH}"

# The ubuntu:24.04 image ships a non-root "ubuntu" user (uid 1000); run as it.
USER ubuntu
