import torch
import numpy as np
from PIL import Image
import io
import requests
import hashlib
import hmac
from datetime import datetime
import os
import re

# =================================================================================
# AWS Signature Version 4 (SigV4) Helper Functions for requests
# No boto3 dependency needed.
# =================================================================================

def _sign(key, msg):
    return hmac.new(key, msg.encode('utf-8'), hashlib.sha256).digest()

def _get_signature_key(key, date_stamp, region_name, service_name):
    """Generates the signature key for AWS SigV4."""
    k_date = _sign(('AWS4' + key).encode('utf-8'), date_stamp)
    k_region = _sign(k_date, region_name)
    k_service = _sign(k_region, service_name)
    k_signing = _sign(k_service, 'aws4_request')
    return k_signing

def _generate_s3_headers(client_config, object_key, payload, content_type='application/octet-stream'):
    """
    Generates the necessary headers for a PUT request to S3,
    including the AWS SigV4 Authorization header.
    UPDATED: Now accepts a dynamic content_type.
    """
    service = 's3'
    method = 'PUT'
    
    access_key = client_config.get('access_key_id')
    secret_key = client_config.get('secret_access_key')
    region = client_config.get('region')
    endpoint_url = client_config.get('endpoint_url')
    bucket_name = client_config.get('bucket_name')

    host_match = re.search(r'https?://([^/]+)', endpoint_url)
    if not host_match:
        raise ValueError("Invalid endpoint_url format. Expected format like 'https://hostname'.")
    host = host_match.group(1)

    t = datetime.utcnow()
    amz_date = t.strftime('%Y%m%dT%H%M%SZ')
    date_stamp = t.strftime('%Y%m%d')

    canonical_uri = f'/{bucket_name}/{object_key}'
    canonical_querystring = ''
    
    payload_hash = hashlib.sha256(payload).hexdigest()
    
    headers = {
        'host': host,
        'x-amz-date': amz_date,
        'x-amz-content-sha256': payload_hash,
    }

    signed_headers = ';'.join(sorted(headers.keys()))
    canonical_headers = ''.join([f'{k}:{v}\n' for k, v in sorted(headers.items())])

    canonical_request = (
        f'{method}\n'
        f'{canonical_uri}\n'
        f'{canonical_querystring}\n'
        f'{canonical_headers}\n'
        f'{signed_headers}\n'
        f'{payload_hash}'
    )

    algorithm = 'AWS4-HMAC-SHA256'
    credential_scope = f'{date_stamp}/{region}/{service}/aws4_request'
    
    string_to_sign = (
        f'{algorithm}\n'
        f'{amz_date}\n'
        f'{credential_scope}\n'
        f'{hashlib.sha256(canonical_request.encode("utf-8")).hexdigest()}'
    )

    signing_key = _get_signature_key(secret_key, date_stamp, region, service)
    signature = hmac.new(signing_key, string_to_sign.encode('utf-8'), hashlib.sha256).hexdigest()

    authorization_header = (
        f'{algorithm} '
        f'Credential={access_key}/{credential_scope}, '
        f'SignedHeaders={signed_headers}, '
        f'Signature={signature}'
    )

    final_headers = {
        'Authorization': authorization_header,
        'x-amz-date': amz_date,
        'x-amz-content-sha256': payload_hash,
        'Content-Type': content_type,
        'Content-Length': str(len(payload))
    }
    
    return final_headers

# =================================================================================
# ComfyUI Custom Nodes
# =================================================================================

class CreateS3Client:
    """
    ComfyUI node to create an S3 client configuration object.
    This object stores credentials and endpoint info for other S3 nodes.
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "endpoint_url": ("STRING", {"default": "https://s3.us-west-004.backblazeb2.com"}),
                "bucket_name": ("STRING", {"default": "my-comfyui-bucket"}),
                "region": ("STRING", {"default": "us-west-004"}),
                "access_key_id": ("STRING", {"default": "YOUR_ACCESS_KEY_ID", "multiline": False}),
                "secret_access_key": ("STRING", {"default": "YOUR_SECRET_ACCESS_KEY", "multiline": False, "input_type": "password"}),
                "public_url_base": ("STRING", {"default": "https://f004.backblazeb2.com/file/my-comfyui-bucket"}),
            }
        }

    RETURN_TYPES = ("S3_CLIENT",)
    FUNCTION = "create_client"
    CATEGORY = "duanyll/web"
    
    def create_client(self, endpoint_url, bucket_name, region, access_key_id, secret_access_key, public_url_base):
        endpoint_url = endpoint_url.rstrip('/')
        public_url_base = public_url_base.rstrip('/')

        s3_client_config = {
            "endpoint_url": endpoint_url,
            "bucket_name": bucket_name,
            "region": region,
            "access_key_id": access_key_id,
            "secret_access_key": secret_access_key,
            "public_url_base": public_url_base
        }
        return (s3_client_config,)

class UploadImageToS3:
    """
    ComfyUI node to upload an image tensor to an S3-compatible storage service
    with options for format and quality.
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "s3_client": ("S3_CLIENT",),
                "image": ("IMAGE",),
                "key": ("STRING", {"default": "comfyui/output/image.png", "multiline": False}),
            },
            "optional": {
                "format": (["png", "jpeg", "webp"], {"default": "png"}),
                "quality": ("INT", {"default": 90, "min": 1, "max": 100, "step": 1}),
                "webp_lossless": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("public_url",)
    FUNCTION = "upload_image"
    CATEGORY = "duanyll/web"
    
    def upload_image(self, s3_client, image, key, format="png", quality=90, webp_lossless=False):
        # 1. Convert torch tensor to a PIL image object
        img_tensor = image[0]
        i = 255. * img_tensor.cpu().numpy()
        img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
        
        # 2. Encode image to bytes based on selected format and quality
        buffer = io.BytesIO()
        mime_type = f"image/{format}"
        save_options = {}

        if format == 'png':
            save_options['compress_level'] = 6 # A good balance for PNG
        elif format == 'jpeg':
            # JPEG does not support alpha channel, convert if necessary
            if img.mode == 'RGBA':
                img = img.convert('RGB')
            save_options['quality'] = quality
            save_options['subsampling'] = 0 # Use 4:4:4 subsampling for higher quality
        elif format == 'webp':
            save_options['quality'] = quality
            save_options['lossless'] = webp_lossless

        try:
            img.save(buffer, format=format.upper(), **save_options)
        except Exception as e:
            print(f"Error saving image to buffer with format {format}: {e}")
            raise

        image_bytes = buffer.getvalue()

        # 3. Automatically adjust the key's extension based on format
        base_key, _ = os.path.splitext(key)
        final_key = f"{base_key}.{format}"
        
        # 4. Generate signed headers for the request
        try:
            headers = _generate_s3_headers(s3_client, final_key, image_bytes, content_type=mime_type)
        except Exception as e:
            print(f"Error generating S3 headers: {e}")
            raise

        # 5. Construct the full URL and perform the upload
        upload_url = f"{s3_client['endpoint_url']}/{s3_client['bucket_name']}/{final_key}"
        
        print(f"Uploading image to S3 ({format.upper()}, quality: {quality}): {upload_url}")
        try:
            response = requests.put(upload_url, data=image_bytes, headers=headers)
            response.raise_for_status()
            print(f"Successfully uploaded image. Status: {response.status_code}")
        except requests.exceptions.RequestException as e:
            print(f"Failed to upload image to S3. Error: {e}")
            if e.response is not None:
                print(f"S3 Response Status: {e.response.status_code}")
                print(f"S3 Response Body: {e.response.text}")
            raise

        # 6. Construct and return the public URL
        public_url = f"{s3_client['public_url_base']}/{final_key}"
        return (public_url,)
