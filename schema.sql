-- =====================================================================
-- MedAssist database schema (MySQL 8+)
-- Run with:  mysql -u root < schema.sql
-- =====================================================================
CREATE DATABASE IF NOT EXISTS medassist
  CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;
USE medassist;

-- A dedicated DB user so the app never uses root (change the password!)
CREATE USER IF NOT EXISTS 'medassist'@'localhost' IDENTIFIED BY 'change_me_123';
GRANT ALL PRIVILEGES ON medassist.* TO 'medassist'@'localhost';
FLUSH PRIVILEGES;

-- 1) users: simple guest-session auth (token is sent in the X-Session-Token header)
CREATE TABLE IF NOT EXISTS users (
  id            INT AUTO_INCREMENT PRIMARY KEY,
  session_token CHAR(64)    NOT NULL,
  username      VARCHAR(80) NULL,
  created_at    TIMESTAMP   NOT NULL DEFAULT CURRENT_TIMESTAMP,
  UNIQUE KEY uq_users_token (session_token)          -- fast token lookup on every request
) ENGINE=InnoDB;

-- 2) documents: one row per uploaded prescription / lab report
CREATE TABLE IF NOT EXISTS documents (
  id            INT AUTO_INCREMENT PRIMARY KEY,
  user_id       INT          NOT NULL,
  doc_type      ENUM('prescription','lab_report') NOT NULL,
  original_name VARCHAR(255) NOT NULL,
  stored_path   VARCHAR(500) NULL,
  raw_ocr_text  MEDIUMTEXT   NULL,
  ocr_engine    VARCHAR(40)  NULL,
  ocr_confidence FLOAT       NULL,
  created_at    TIMESTAMP    NOT NULL DEFAULT CURRENT_TIMESTAMP,
  CONSTRAINT fk_docs_user FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
  INDEX idx_docs_user_created (user_id, created_at),
  INDEX idx_docs_type (doc_type)
) ENGINE=InnoDB;

-- 3) analysis_results: the AI (or rule-based) output for a document
CREATE TABLE IF NOT EXISTS analysis_results (
  id             INT AUTO_INCREMENT PRIMARY KEY,
  document_id    INT NOT NULL,
  condition_text VARCHAR(255) NULL,
  medicines_json JSON NULL,
  dos_donts_json JSON NULL,
  lab_flags_json JSON NULL,
  summary_text   TEXT NULL,
  source         ENUM('llm','rule_based') NOT NULL DEFAULT 'llm',
  created_at     TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
  CONSTRAINT fk_analysis_doc FOREIGN KEY (document_id) REFERENCES documents(id) ON DELETE CASCADE,
  UNIQUE KEY uq_analysis_doc (document_id)
) ENGINE=InnoDB;

-- 4) chat_history: every chat turn; document_id is NULL for general chat
CREATE TABLE IF NOT EXISTS chat_history (
  id          INT AUTO_INCREMENT PRIMARY KEY,
  user_id     INT NOT NULL,
  document_id INT NULL,
  role        ENUM('user','assistant') NOT NULL,
  message     TEXT NOT NULL,
  created_at  TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
  CONSTRAINT fk_chat_user FOREIGN KEY (user_id)     REFERENCES users(id)     ON DELETE CASCADE,
  CONSTRAINT fk_chat_doc  FOREIGN KEY (document_id) REFERENCES documents(id) ON DELETE SET NULL,
  INDEX idx_chat_lookup (user_id, document_id, created_at)
) ENGINE=InnoDB;
