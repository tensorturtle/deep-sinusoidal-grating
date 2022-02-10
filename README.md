# Deep Sinusoidal Grating

[![Start web app (sine.tensorturtle.com)](https://github.com/tensorturtle/deep-sinusoidal-grating/actions/workflows/start_web_app.yml/badge.svg)](https://github.com/tensorturtle/deep-sinusoidal-grating/actions/workflows/start_web_app.yml)

Exploratory analysis for insights into ML interpretability using well-studied psychological stimuli.

## Quickstart

Start with the no-code beginner's tutorial at https://sine.tensorturtle.com.

## TODO

- [x] Validation per epoch
- [x] ipywidgets modification for distribution parameters
- [ ] Training accuracy Visualization 
- [ ] Training regime similar to humans (try with starting with pre-trained?)
- [x] Convolution visualization integration: https://jacobgil.github.io/deeplearning/filter-visualizations, https://github.com/fossasia/visdom
- [x] Hosting on blog solution
- [x] Count number of parameters
- [x] Saveable dataset

## Developing No-Code Tutorial (Sinusoidal Gratings Generator)

### Writing

Install voila

```
pip install voila
```

Download [`retro` theme](https://github.com/martinRenou/voila-retro)

```
pip install voila-retro
```

In the root directory of this repo:

```
voila no_code_turorial.ipynb --template=retro
```

### Deployment

Caddyfile at /etc/caddy/Caddyfile
```
sine.tensorturtle.com {
  reverse_proxy localhost:19999
}
```

Systemd service at /etc/systemd/system/voila.service

```
[Unit]
Description=Voila

[Service]
Type=simple
PIDFile=/run/voila.pid
ExecStart=/usr/bin/python3 -m voila --no-browser --template=retro --port=19999 --show_tracebacks=True --preheat_kernel=True --pool_size=3 /home/deploy/voila/no_code_tutorial.ipynb
User=deploy
WorkingDirectory=/home/deploy/
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

preheat_kernel massively reduces the loading time, at least for the first pool_size connections.

SCP the contents of this repo into /home/deploy/voila

`sudo apt install libjpeg-dev zlib1g-dev` to install the necessary dependencies for Pillow

```
python3 -m pip install -r requirements.txt
```

Install pytorch from https://pytorch.org. On a system with <4GB of RAM, you might need to increase swap size.
