已完成 openpi_RTC 下 JAX 模式 pi05 + RTC 真机前离线测试，未修改任何文档或代码。

  这次实际跑通的是：
  parquet 离线观测 -> [serve_rtc_pi0.py](/share/project/Ruimeng/openpi_RTC/examples/real_robot_rtc/
  serve_rtc_pi0.py) -> [websocket_server.py](/share/project/Ruimeng/openpi_RTC/examples/
  real_robot_rtc/websocket_server.py) -> policy.infer(JAX) -> websocket client 回收 action chunk

  关键结果：

  - 本地 /share/project/Ruimeng/openpi_RTC/.venv 的 JAX 可用，但 PyAV 缺少 libavformat.so.61，不能直
    接用于这条链路
  - 参考运行时可用：/share/project/wujiling/openpi/.venv + PYTHONHOME=/share/project/wujiling/
    envs/.uvpy/python/cpython-3.11.13-linux-x86_64-gnu
  - 需要强制让 Python 导入当前仓库代码，而不是 wujiling/openpi 那份旧代码，所以还要加：

  export PYTHONPATH=/share/project/Ruimeng/openpi_RTC/src:/share/project/Ruimeng/openpi_RTC/packages/
  openpi-client/src${PYTHONPATH:+:$PYTHONPATH}

  - 可用 checkpoint：
    /share/project/wujiling/checkpoints/finetune/pi05_pnp_orange/20000
  - 可用离线数据：
    /share/project/section/task/orange_200_9_8/data/chunk-000/episode_000003.parquet

  我验证了两层：

  - 直接单样本 policy.infer 成功，返回 actions 形状为 (50, 14)
  - RTC websocket 全链路连续跑 4 帧成功，每次都收到 (50, 14) 的 action chunk

  这次实际使用的 server 启动命令是：

  source /share/project/wujiling/openpi/.venv/bin/activate
  export ROOT=/share/project/wujiling/envs/.uvpy/python/cpython-3.11.13-linux-x86_64-gnu
  export PYTHONHOME="$ROOT"
  export PYTHONPATH=/share/project/Ruimeng/openpi_RTC/src:/share/project/Ruimeng/openpi_RTC/packages/
  openpi-client/src${PYTHONPATH:+:$PYTHONPATH}
  cd /share/project/Ruimeng/openpi_RTC/examples/real_robot_rtc
  python serve_rtc_pi0.py --port 8000 policy:checkpoint --policy.config pi05_agilex_orange
  --policy.dir /share/project/wujiling/checkpoints/finetune/pi05_pnp_orange/20000

  结论是：pi05 + RTC 的离线链路本身是通的，当前真机前最大的环境注意点只有一个，别用 openpi_RTC/.venv
  直接跑这条链路，要用上面那套运行时和 PYTHONPATH 覆盖。