# Use your training image as the base
FROM 114.214.255.71:5000/training/training:v1.1

# Install SSH server
RUN apt-get update && apt-get install -y openssh-server sudo

# Create SSH runtime directory
RUN mkdir /var/run/sshd

# Set root password
RUN echo "root:your_password_here" | chpasswd

# Allow root login
RUN sed -i 's/#*PermitRootLogin.*/PermitRootLogin yes/' /etc/ssh/sshd_config

# Install Python packages
RUN pip install --no-cache-dir \
    asttokens==3.0.1 \
    certifi==2026.6.17 \
    chardet==4.0.0 \
    colorama==0.4.6 \
    comm==0.2.3 \
    contourpy==1.3.3 \
    cycler==0.12.1 \
    debugpy==1.8.21 \
    decorator==5.3.1 \
    executing==2.2.1 \
    filelock==3.29.4 \
    fonttools==4.63.0 \
    fsspec==2026.6.0 \
    idna==2.10 \
    ipykernel==7.3.0 \
    ipython==9.15.0 \
    ipython_pygments_lexers==1.1.1 \
    jedi==0.20.0 \
    Jinja2==3.1.6 \
    jupyter_client==8.9.1 \
    jupyter_core==5.9.1 \
    kiwisolver==1.5.0 \
    MarkupSafe==3.0.3 \
    matplotlib==3.11.0 \
    matplotlib-inline==0.2.2 \
    mpmath==1.3.0 \
    nest-asyncio2==1.7.2 \
    networkx==3.6.1 \
    numpy==2.5.0 \
    packaging==26.2 \
    pandas==3.0.3 \
    parso==0.8.7 \
    pillow==12.2.0 \
    platformdirs==4.10.0 \
    prompt_toolkit==3.0.52 \
    psutil==7.2.2 \
    pure_eval==0.2.3 \
    Pygments==2.20.0 \
    pyparsing==3.3.2 \
    python-dateutil==2.9.0.post0 \
    pyzmq==27.1.0 \
    requests==2.25.1 \
    scipy==1.18.0 \
    seaborn==0.13.2 \
    setuptools==81.0.0 \
    six==1.17.0 \
    stack-data==0.6.3 \
    sympy==1.14.0 \
    torch==2.12.1 \
    torchao==0.17.0 \
    torchvision==0.27.1 \
    tornado==6.5.7 \
    tqdm==4.68.3 \
    traitlets==5.15.1 \
    typing_extensions==4.15.0 \
    tzdata==2026.2 \
    urllib3==1.26.20 \
    wcwidth==0.8.2

# Install autoattack separately (since it's from git)
RUN pip install --no-cache-dir git+https://github.com/fra31/auto-attack@a39220048b3c9f2cca9a4d3a54604793c68eca7e

# Expose SSH port
EXPOSE 22

# Start SSH service when container runs
CMD ["/usr/sbin/sshd", "-D"]