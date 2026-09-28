CREATE DATABASE IF NOT EXISTS cancer_detection_project_db;
USE cancer_detection_project_db;

CREATE TABLE IF NOT EXISTS users (
    id INT AUTO_INCREMENT PRIMARY KEY,
    username VARCHAR(100) NOT NULL UNIQUE,
    -- password hashes are ~100-170 characters long
    password VARCHAR(255) NOT NULL
);

-- For an existing database, widen the password column so hashes are not truncated:
-- ALTER TABLE users MODIFY password VARCHAR(255) NOT NULL;
