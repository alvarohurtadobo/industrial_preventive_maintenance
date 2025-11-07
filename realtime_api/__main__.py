import uvicorn

from .app import get_settings


def main() -> None:
    settings = get_settings()
    uvicorn.run(
        "realtime_api.app:app",
        host=settings.host,
        port=settings.port,
        reload=settings.reload,
    )


if __name__ == "__main__":
    main()

