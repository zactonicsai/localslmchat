#!/bin/bash
echo "Creating S3 bucket: lcq-documents"
awslocal s3 mb s3://lcq-documents
echo "S3 bucket created."
