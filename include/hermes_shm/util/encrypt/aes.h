//
// Created by lukemartinlogan on 2/4/24.
//

#ifndef HERMES_SHM_INCLUDE_HERMES_SHM_UTIL_ENCRYPT_AES_H_
#define HERMES_SHM_INCLUDE_HERMES_SHM_UTIL_ENCRYPT_AES_H_

#include <openssl/evp.h>
#include <openssl/aes.h>
#include <openssl/rand.h>

#include <string>
#include <hermes_shm/data_structures/data_structure.h>


class AES {
 public:
  std::string key_;
  std::string iv_;

 public:
  void GenerateKey(const std::string &password, uint32_t bytes,
                   const std::string &salt="") {
    const EVP_CIPHER *cipher = EVP_aes_256_cbc();
    const EVP_MD *digest = EVP_sha256();
    key_ = std::string(bytes, 0);
    iv_ = std::string(EVP_CIPHER_iv_length(cipher), 0);
    int ret = EVP_BytesToKey(cipher, digest,
                             (unsigned char*)salt.c_str(),
                             (unsigned char*)password.c_str(),
                             password.size(), 1,
                             (unsigned char*)key_.c_str(),
                             (unsigned char*)iv_.c_str());
    if (!ret) {
      HELOG(kError, "Failed to generate key");
    }
  }

  bool Encrypt(char *output, size_t &output_size, char *input, size_t input_size) {
    EVP_CIPHER_CTX *ctx;
    int ret;

    if (!(ctx = EVP_CIPHER_CTX_new()))
      return false;

    ret = EVP_EncryptInit_ex(ctx, EVP_aes_256_cbc(), NULL,
                             (unsigned char*)key_.c_str(),
                             (unsigned char*)iv_.c_str());
    if (1 != ret)
      return false;

    int output_len_int = input_size;
    ret =  EVP_EncryptUpdate(ctx,
                             (unsigned char*)output,
                             (int*)&output_len_int,
                             (unsigned char*)input,
                             input_size);
    if (1 != ret)
      return false;

    int ciphertext_len;
    ret = EVP_EncryptFinal_ex(ctx,
                              (unsigned char*)output + input_size,
                              &ciphertext_len);
    output_size = input_size + ciphertext_len;
    if (1 != ret)
      return false;

    EVP_CIPHER_CTX_free(ctx);
    return true;
  }

  bool Decrypt(char *output, size_t &output_size,
               char *input, size_t input_size) {
    EVP_CIPHER_CTX *ctx;
    int ret;

    if (!(ctx = EVP_CIPHER_CTX_new()))
      return false;

    ret = EVP_DecryptInit_ex(ctx, EVP_aes_256_cbc(), NULL,
                             (unsigned char*)key_.c_str(),
                             (unsigned char*)iv_.c_str());
    if (1 != ret)
      return false;

    int output_size_int;
    ret = EVP_DecryptUpdate(
        ctx,
        (unsigned char*)output, &output_size_int,
        (unsigned char*)input, input_size);
    if (1 != ret)
      return false;
    output_size = output_size_int;


    int plaintext_len;
    ret = EVP_DecryptFinal_ex(
        ctx, (unsigned char*)output + output_size_int, &plaintext_len);
    if (1 != ret)
      return false;

    EVP_CIPHER_CTX_free(ctx);
    return true;
  }
};

#endif //HERMES_SHM_INCLUDE_HERMES_SHM_UTIL_ENCRYPT_AES_H_
