use actix_files::Files;
use actix_web::{App, HttpServer};

#[actix_web::main]
async fn main() -> std::io::Result<()> {
    let addr = "127.0.0.1:6161";
    println!("Server started at http://{}", addr);

    HttpServer::new(|| {
        App::new()
            // Serve index.html and other static files from the current directory
            .service(Files::new("/", ".").index_file("index.html"))
    })
    .bind(addr)?
    .run()
    .await
}