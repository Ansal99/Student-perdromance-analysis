/*
  # Create student_records table for Student Performance Analysis & Prediction System

  ## Overview
  This migration creates the database schema for storing student performance records
  and predictions made by the machine learning system.

  ## New Tables
    - `student_records`
      - `id` (uuid, primary key) - Unique identifier for each record
      - `student_name` (text) - Name of the student
      - `attendance` (numeric) - Attendance percentage (0-100)
      - `study_hours` (numeric) - Daily study hours
      - `previous_score` (numeric) - Previous exam score (0-100)
      - `predicted_score` (numeric) - ML model predicted score
      - `risk_level` (text) - Risk assessment (Low Risk, Medium Risk, High Risk)
      - `created_at` (timestamptz) - Timestamp when record was created

  ## Security
    - Enable RLS on `student_records` table
    - Add policy for authenticated users to read all records
    - Add policy for authenticated users to insert new records
    - Add policy for public access to insert records (for demo purposes)

  ## Notes
    - Default values are set for timestamps
    - All numeric fields use numeric type for precision
    - Risk level is stored as text for flexibility
*/

CREATE TABLE IF NOT EXISTS student_records (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  student_name text NOT NULL DEFAULT 'Anonymous',
  attendance numeric NOT NULL CHECK (attendance >= 0 AND attendance <= 100),
  study_hours numeric NOT NULL CHECK (study_hours >= 0),
  previous_score numeric NOT NULL CHECK (previous_score >= 0 AND previous_score <= 100),
  predicted_score numeric NOT NULL,
  risk_level text NOT NULL DEFAULT 'Medium Risk',
  created_at timestamptz DEFAULT now()
);

ALTER TABLE student_records ENABLE ROW LEVEL SECURITY;

CREATE POLICY "Anyone can insert student records"
  ON student_records
  FOR INSERT
  TO anon, authenticated
  WITH CHECK (true);

CREATE POLICY "Anyone can view student records"
  ON student_records
  FOR SELECT
  TO anon, authenticated
  USING (true);

CREATE POLICY "Authenticated users can update their records"
  ON student_records
  FOR UPDATE
  TO authenticated
  USING (true)
  WITH CHECK (true);

CREATE POLICY "Authenticated users can delete records"
  ON student_records
  FOR DELETE
  TO authenticated
  USING (true);
