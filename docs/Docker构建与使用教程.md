# Docker 构建与使用教程

> 适用项目：`D:\rag_agentic`
>
> 编写日期：2026-08-17
>
> 运行环境：Windows、PowerShell、Docker Desktop（Linux containers）
>
> 当前仓库状态：Docker Compose 负责 MySQL、Redis、etcd、MinIO、Milvus；Python API 和 Vue 目前仍在宿主机运行。

## 1. 先理解 Docker 中的几个对象

### 1.1 Image：镜像

镜像是一个只读的软件运行模板，包含操作系统基础层、依赖和程序文件。例如：

```text
mysql:8.4
redis:7.4
milvusdb/milvus:v2.5.27
```

镜像本身不是正在运行的服务。

### 1.2 Container：容器

容器是镜像的一次运行实例。一个镜像可以创建多个容器。

```text
mysql:8.4 镜像
    ↓ docker run / docker compose up
rag-agentic-mysql 容器
```

### 1.3 Volume：数据卷

容器删除后，容器内部的临时文件通常也会消失。Volume 用于独立保存数据库数据。

本项目定义了：

```text
mysql_data
redis_data
etcd_data
minio_data
milvus_data
```

因此，普通的 `docker compose down` 不会删除数据库数据；`docker compose down --volumes` 会删除这些数据卷。

### 1.4 Network：容器网络

Compose 会自动创建项目网络。容器之间使用服务名通信：

```text
mysql:3306
redis:6379
milvus:19530
etcd:2379
minio:9000
```

宿主机上的 Python API 则使用：

```text
127.0.0.1:3306
127.0.0.1:6379
127.0.0.1:19530
```

注意：如果以后把 Python API 也放进容器，容器里的 `127.0.0.1` 只代表 API 容器自身，不能再用它连接数据库，必须改用 Compose 服务名。

### 1.5 Dockerfile 与 Docker Compose

- `Dockerfile`：描述“怎样构建一个自己的镜像”。
- `docker-compose.yml`：描述“运行哪些服务，以及它们如何连接”。

本项目当前的 `docker-compose.yml` 全部使用 `image:`，没有 `build:`：

```yaml
services:
  mysql:
    image: mysql:8.4
```

所以当前执行：

```powershell
docker compose pull
docker compose up -d
```

是在下载并运行官方镜像，不是在构建本项目 Python 代码。

## 2. 当前项目的 Docker 架构

```text
Windows 宿主机
├── Query API      127.0.0.1:8000    uv run uvicorn
├── Admin API      127.0.0.1:8001    uv run uvicorn
├── Vue            127.0.0.1:5173    npm run dev
├── 本地 BGE 模型
└── Docker Desktop
    ├── MySQL       127.0.0.1:3306
    ├── Redis       127.0.0.1:6379
    ├── Milvus      127.0.0.1:19530
    ├── Milvus 健康 127.0.0.1:9091
    ├── etcd        仅容器网络
    └── MinIO       仅容器网络
```

这是一种适合本地开发的“应用跑宿主机、基础设施跑 Docker”模式。

## 3. Windows 安装和启动检查

### 3.1 安装 Docker Desktop

安装 Docker Desktop，并确认使用 Linux containers。Windows 通常使用 WSL 2 后端。

安装后先启动 Docker Desktop，等待界面显示 Docker Engine 正常运行。

### 3.2 检查命令

在 PowerShell 中运行：

```powershell
docker --version
docker compose version
docker info
```

前两个命令只能证明 CLI 已安装。`docker info` 成功，才能证明 Docker Engine 可以连接。

如果看到类似错误：

```text
open //./pipe/dockerDesktopLinuxEngine: The system cannot find the file specified
```

通常表示 Docker Desktop 没有启动，或者当前没有运行 Linux Engine。

## 4. 第一次启动本项目基础设施

### 4.1 进入项目根目录

```powershell
Set-Location D:\rag_agentic
```

后面的 Compose 命令默认都应在包含 `docker-compose.yml` 的根目录执行。

### 4.2 创建本地 `.env`

仅在 `.env` 不存在时执行：

```powershell
Copy-Item .env.example .env
```

然后编辑：

```powershell
notepad .env
```

至少替换：

```text
MYSQL_PASSWORD
MYSQL_ROOT_PASSWORD
REDIS_PASSWORD
MILVUS_PASSWORD
MINIO_ACCESS_KEY
MINIO_SECRET_KEY
```

注意：

- 不要把真实密钥写入 `docker-compose.yml`。
- 不要提交 `.env`。
- `.env` 已被 `.gitignore` 忽略。
- `.env.example` 只能放占位符。
- `REWRITE_LLM_BASE_URL=http://127.0.0.1:8000/v1` 与本项目 Query API 默认端口冲突，使用外部 OpenAI 兼容模型时必须改成真实模型地址。

可以检查忽略规则：

```powershell
git check-ignore -v .env
```

### 4.3 静默校验 Compose 配置

```powershell
docker compose config -q
```

成功时通常没有输出，退出码为 `0`。

不要随意把下面命令的完整结果发到公开聊天或日志：

```powershell
docker compose config
```

因为它会展开 `.env` 中的变量，输出中可能出现数据库密码。

### 4.4 下载镜像

```powershell
docker compose pull
```

本项目会下载：

```text
mysql:8.4
redis:7.4
quay.io/coreos/etcd:v3.5.18
minio/minio:RELEASE.2025-02-18T16-25-55Z
milvusdb/milvus:v2.5.27
```

Milvus 镜像较大，第一次下载可能需要较长时间。

### 4.5 启动并等待健康检查

推荐：

```powershell
docker compose up -d --wait
```

参数含义：

- `up`：创建并启动服务。
- `-d`：后台运行，不持续占用当前终端。
- `--wait`：等待服务达到 running 或 healthy 状态。

如果不使用 `--wait`：

```powershell
docker compose up -d
docker compose ps
```

### 4.6 查看状态

```powershell
docker compose ps
```

应重点检查：

```text
mysql     healthy
redis     healthy
etcd      healthy
minio     healthy
milvus    healthy
```

只有容器名称存在，不代表服务可用；应等待健康状态。

## 5. 验证每个服务

### 5.1 验证 MySQL

```powershell
docker compose exec mysql sh -c 'mysqladmin ping -h localhost -uroot -p"$MYSQL_ROOT_PASSWORD"'
```

期望看到：

```text
mysqld is alive
```

### 5.2 验证 Redis

```powershell
docker compose exec redis sh -c 'redis-cli -a "$REDIS_PASSWORD" ping'
```

期望看到：

```text
PONG
```

### 5.3 验证 Milvus

```powershell
Invoke-RestMethod http://127.0.0.1:9091/healthz
```

也可以查看容器状态：

```powershell
docker compose ps milvus
```

### 5.4 检查宿主机端口

```powershell
Get-NetTCPConnection -State Listen -LocalPort 3306,6379,19530,9091
```

如果某个端口没有监听，先查对应服务日志。

## 6. 启动项目应用

当前 Python API 和 Vue 不在 Docker 中。基础设施健康后，在三个 PowerShell 窗口分别运行。

### 6.1 Admin API

```powershell
Set-Location D:\rag_agentic
uv run uvicorn admin_api.main:app --host 127.0.0.1 --port 8001
```

接口文档：

```text
http://127.0.0.1:8001/docs
```

### 6.2 Query API

```powershell
Set-Location D:\rag_agentic
uv run uvicorn query_api.main:app --host 127.0.0.1 --port 8000
```

接口文档：

```text
http://127.0.0.1:8000/docs
```

### 6.3 Vue 工作台

```powershell
Set-Location D:\rag_agentic\frontend
npm.cmd install
npm.cmd run dev
```

页面：

```text
http://127.0.0.1:5173
```

管理 API 当前没有鉴权，只能绑定到 `127.0.0.1`，不要改成 `0.0.0.0` 后直接暴露到局域网或公网。

## 7. 日常使用命令

### 7.1 查看全部容器

```powershell
docker compose ps
```

### 7.2 查看全部日志

```powershell
docker compose logs
```

### 7.3 持续跟踪日志

```powershell
docker compose logs -f
```

按 `Ctrl+C` 只会退出日志跟踪，后台容器仍会继续运行。

### 7.4 查看单个服务日志

```powershell
docker compose logs -f milvus
docker compose logs -f mysql
docker compose logs -f redis
```

### 7.5 停止服务但保留容器

```powershell
docker compose stop
```

重新启动：

```powershell
docker compose start
```

### 7.6 重启服务

```powershell
docker compose restart
```

重启单个服务：

```powershell
docker compose restart redis
```

### 7.7 停止并删除容器，保留数据卷

```powershell
docker compose down
```

该命令会删除 Compose 容器和默认网络，但默认保留命名数据卷。之后再执行 `up`，数据库数据仍应存在。

### 7.8 删除容器和全部数据卷

```powershell
docker compose down --volumes
```

警告：该命令会删除 MySQL、Redis、Milvus、MinIO、etcd 的持久化数据。只有在明确需要完全重建本地数据，并且已确认不需要备份时才能执行。

不要为了修复普通启动错误直接使用 `down --volumes`。

## 8. 只操作一个或部分服务

启动 MySQL 和 Redis：

```powershell
docker compose up -d mysql redis
```

启动 Milvus 时，还需要它依赖的 etcd 和 MinIO。Compose 会根据 `depends_on` 自动启动依赖：

```powershell
docker compose up -d --wait milvus
```

停止一个服务：

```powershell
docker compose stop redis
```

查看一个服务：

```powershell
docker compose ps redis
```

## 9. 更新镜像

先查看当前状态：

```powershell
docker compose ps
```

下载 Compose 文件指定版本的最新镜像：

```powershell
docker compose pull
```

重新创建服务并保留数据卷：

```powershell
docker compose up -d --wait
```

项目已经固定了具体 Milvus 和 MinIO 版本，不应在未验证兼容性的情况下直接改成 `latest`。

## 10. Dockerfile 是怎样构建镜像的

常见指令：

| 指令 | 作用 |
| --- | --- |
| `FROM` | 选择基础镜像 |
| `WORKDIR` | 设置容器内工作目录 |
| `COPY` | 把构建上下文中的文件复制进镜像 |
| `RUN` | 构建镜像时执行命令 |
| `ENV` | 设置镜像或容器环境变量 |
| `EXPOSE` | 声明应用监听端口，不等于映射端口 |
| `CMD` | 容器默认启动命令 |
| `ENTRYPOINT` | 定义固定入口程序 |

示例：

```dockerfile
FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["uvicorn", "query_api.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

构建：

```powershell
docker build -f Dockerfile.api -t rag-agentic-api:0.1.0 .
```

各参数：

```text
-f Dockerfile.api          指定 Dockerfile
-t rag-agentic-api:0.1.0   设置镜像名和标签
.                          当前目录是构建上下文
```

查看镜像：

```powershell
docker image ls rag-agentic-api
```

运行一个临时示例：

```powershell
docker run --rm -p 127.0.0.1:8000:8000 rag-agentic-api:0.1.0
```

注意：上述 Dockerfile 是构建原理示例，当前仓库尚未创建 `Dockerfile.api`，也没有完成 API 容器的真实验收，不能把它当作已交付功能。

## 11. 为本项目构建 Python API 时需要额外解决的问题

### 11.1 容器内数据库地址

API 容器不能使用：

```text
MYSQL_HOST=127.0.0.1
REDIS_HOST=127.0.0.1
MILVUS_HOST=127.0.0.1
```

应覆盖为：

```text
MYSQL_HOST=mysql
REDIS_HOST=redis
MILVUS_HOST=milvus
```

### 11.2 本地模型

仓库中的模型目录已被 Git 忽略：

```text
models/bge-m3
models/bge-reranker-large
models/bert-base-chinese
model_trian_classify/model/best_model
```

不建议把大型模型直接复制进普通应用镜像。开发环境可挂载只读目录：

```yaml
volumes:
  - ./models:/app/models:ro
  - ./model_trian_classify/model:/app/model_trian_classify/model:ro
```

生产环境可以使用独立模型镜像、对象存储下载步骤或模型服务。

### 11.3 GPU

当前代码会在可用时使用 `cuda:0`，依赖配置还指定了 CUDA 版 PyTorch。构建出的镜像可能很大。

GPU 容器化前必须明确：

- 宿主机 NVIDIA 驱动是否可用；
- Docker Desktop 的 WSL 2 GPU 支持是否正常；
- 容器 PyTorch CUDA 版本是否与驱动兼容；
- Compose 是否为服务配置 GPU；
- Query API 和 Admin API 同时加载模型时的显存占用。

不要在没有 GPU 容器验收的情况下声称镜像支持 CUDA。

### 11.4 上传目录

Admin API 的 `uploads/` 必须使用持久化卷或宿主机挂载，否则重建容器后上传原文件会丢失。

例如：

```yaml
volumes:
  - ./uploads:/app/uploads
```

### 11.5 安全边界

容器里的 Uvicorn 必须监听 `0.0.0.0` 才能被宿主机端口映射访问，但宿主机端口仍应绑定回环地址：

```yaml
ports:
  - "127.0.0.1:8000:8000"
```

这两个地址的含义不同：

```text
容器内 0.0.0.0       接受来自容器网络的连接
宿主机 127.0.0.1     只允许本机访问映射端口
```

## 12. 推荐的完整容器化 Compose 形态

以下仅用于说明未来结构，不是当前仓库已经实现的配置：

```yaml
services:
  query-api:
    build:
      context: .
      dockerfile: Dockerfile.api
    command: ["uvicorn", "query_api.main:app", "--host", "0.0.0.0", "--port", "8000"]
    env_file:
      - .env
    environment:
      MYSQL_HOST: mysql
      REDIS_HOST: redis
      MILVUS_HOST: milvus
    ports:
      - "127.0.0.1:8000:8000"
    volumes:
      - ./models:/app/models:ro
      - ./model_trian_classify/model:/app/model_trian_classify/model:ro
    depends_on:
      mysql:
        condition: service_healthy
      redis:
        condition: service_healthy
      milvus:
        condition: service_healthy

  admin-api:
    build:
      context: .
      dockerfile: Dockerfile.api
    command: ["uvicorn", "admin_api.main:app", "--host", "0.0.0.0", "--port", "8001"]
    env_file:
      - .env
    environment:
      MYSQL_HOST: mysql
      REDIS_HOST: redis
      MILVUS_HOST: milvus
    ports:
      - "127.0.0.1:8001:8001"
    volumes:
      - ./models:/app/models:ro
      - ./uploads:/app/uploads
    depends_on:
      mysql:
        condition: service_healthy
      redis:
        condition: service_healthy
      milvus:
        condition: service_healthy
```

实现后可以使用：

```powershell
docker compose build query-api admin-api
docker compose up -d --wait
```

或者：

```powershell
docker compose up -d --build --wait
```

## 13. `.dockerignore`

构建自己的镜像前应创建 `.dockerignore`，避免把密钥、虚拟环境、缓存和大量本地模型发送到 Docker 构建上下文。

建议内容：

```dockerignore
.git
.env
.venv
__pycache__
.pytest_cache
.mypy_cache
.ruff_cache
.uv-cache
frontend/node_modules
frontend/dist
log
uploads
volumes
models
model_trian_classify/model
```

如果某个模型必须被复制进镜像，应显式调整规则；不要为了方便直接删除所有忽略项。

## 14. 镜像构建缓存

Dockerfile 中先复制依赖文件，再复制业务代码：

```dockerfile
COPY requirements.txt ./
RUN pip install -r requirements.txt
COPY . .
```

这样普通 Python 代码变化时，可以复用依赖安装层。

完全禁用缓存重新构建：

```powershell
docker build --no-cache -f Dockerfile.api -t rag-agentic-api:0.1.0 .
```

不要把 `--no-cache` 当成常规命令；它会重新下载和安装全部依赖。

## 15. 常见问题排查

### 15.1 Docker Engine 无法连接

症状：

```text
The system cannot find the file specified
dockerDesktopLinuxEngine
```

处理顺序：

1. 打开 Docker Desktop。
2. 确认处于 Linux containers。
3. 等待 Engine 启动完成。
4. 运行 `docker info`。
5. 再运行 `docker compose ps`。

### 15.2 端口已被占用

检查：

```powershell
Get-NetTCPConnection -State Listen -LocalPort 3306,6379,19530,9091
```

查看进程：

```powershell
Get-Process -Id (Get-NetTCPConnection -State Listen -LocalPort 3306).OwningProcess
```

不要在不知道进程用途时直接结束它。可以先停止旧数据库服务，或明确调整 Compose 的宿主机映射端口，并同步修改 `.env`。

### 15.3 容器一直 unhealthy

```powershell
docker compose ps
docker compose logs --tail 200 mysql
docker compose logs --tail 200 redis
docker compose logs --tail 200 etcd
docker compose logs --tail 200 minio
docker compose logs --tail 200 milvus
```

先修复最早失败的依赖。Milvus 依赖 etcd 和 MinIO，不能只看 Milvus 最后一条报错。

### 15.4 修改 `.env` 密码后 MySQL 无法登录

MySQL 初始化密码通常只在空数据目录第一次启动时生效。已有 `mysql_data` 时，修改 `.env` 不会自动修改数据库内部账号密码。

可选处理：

- 使用原密码进入 MySQL，再执行密码变更；
- 或在确认数据不再需要后删除数据卷并重新初始化。

不要未经备份直接执行 `docker compose down --volumes`。

### 15.5 Milvus 连接被拒绝

按顺序检查：

```powershell
docker compose ps
docker compose logs --tail 200 etcd
docker compose logs --tail 200 minio
docker compose logs --tail 200 milvus
Invoke-RestMethod http://127.0.0.1:9091/healthz
Get-NetTCPConnection -State Listen -LocalPort 19530
```

不要因为目录或镜像存在就推断 Milvus 已经可用。

### 15.6 镜像下载超时

再次执行：

```powershell
docker compose pull
```

Docker 会复用已经下载的层。仍失败时再检查 Docker Desktop 代理、DNS、网络连接和可用磁盘空间。

## 16. 安全和数据边界

- 所有宿主机端口目前都绑定 `127.0.0.1`，不要随意改成 `0.0.0.0`。
- 不把 `.env`、API Key、数据库密码构建进镜像。
- 不使用 `latest` 替代已经验证的数据库和 Milvus 版本。
- 上传目录和数据库卷需要明确备份策略。
- 删除容器不等于删除数据卷。
- `docker compose down --volumes` 是数据删除操作。
- 不把 Docker socket 挂载给普通应用容器。
- Python API 完整容器化之前，需要先实现接口鉴权、提示词安全边界和健康检查。

## 17. 推荐的日常命令链

### 开发开始

```powershell
Set-Location D:\rag_agentic
docker compose config -q
docker compose up -d --wait
docker compose ps
```

然后分别启动 Admin API、Query API 和 Vue。

### 查看故障

```powershell
Set-Location D:\rag_agentic
docker compose ps
docker compose logs --tail 200
```

### 当天开发结束但保留容器

```powershell
docker compose stop
```

### 删除容器但保留数据

```powershell
docker compose down
```

### 完全重建数据

先确认备份和目标范围，再执行：

```powershell
docker compose down --volumes
docker compose pull
docker compose up -d --wait
```

## 18. 当前结论

本项目现在已经具备可用的基础设施 Compose：

```text
MySQL + Redis + etcd + MinIO + Milvus
```

当前正确使用方式是：

```text
Compose 启动基础设施
宿主机 uv 启动两个 Python API
宿主机 npm 启动 Vue
```

目前还不能称为“整个应用都完成 Docker 构建”，因为仓库尚未提供：

```text
Python API Dockerfile
Vue Dockerfile
.dockerignore
API health/readiness
应用服务 Compose 配置
GPU 容器验收
接口鉴权和安全边界
```

后续若要完整容器化，应把它作为独立功能实现、测试和验收，而不是只在 Compose 中临时增加几行配置。

## 19. 官方参考资料

- [Docker Build](https://docs.docker.com/build/)
- [Dockerfile 构建最佳实践](https://docs.docker.com/build/building/best-practices/)
- [多阶段构建](https://docs.docker.com/build/building/multi-stage/)
- [Docker Compose](https://docs.docker.com/compose/)
- [Docker Compose Quickstart](https://docs.docker.com/compose/gettingstarted/)
- [`docker compose up` 命令参考](https://docs.docker.com/reference/cli/docker/compose/up/)
