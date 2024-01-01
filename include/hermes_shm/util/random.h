//
// Created by lukemartinlogan on 1/1/24.
//

#ifndef HERMES_SHM_INCLUDE_HERMES_SHM_UTIL_RANDOM_H_
#define HERMES_SHM_INCLUDE_HERMES_SHM_UTIL_RANDOM_H_

#include <memory>
#include <random>
#include <memory>
#include <chrono>

namespace hshm {

class Distribution {
 protected:
  std::default_random_engine generator;

 public:
  void Seed() {
    generator = std::default_random_engine(
        std::chrono::steady_clock::now().time_since_epoch().count());
  }
  void Seed(size_t seed) { 
    generator = std::default_random_engine(seed);
  }
};

class CountDistribution : public Distribution {
 private:
  size_t inc_ = 1;
  size_t count_ = 0;
 public:
  virtual void Shape(size_t inc) { inc_ = inc; }
  virtual int GetInt() { int temp = count_; count_+=inc_; return temp; }
  virtual size_t GetSize() { size_t temp = count_; count_+=inc_; return temp; };
  virtual double GetDouble() { double temp = count_; count_+=inc_; return temp; };
};

class NormalDistribution : public Distribution {
 private:
  std::normal_distribution<double> distribution_;
  //TODO: add binomial dist
 public:
  NormalDistribution() = default;
  void Shape(double std) {
    distribution_ = std::normal_distribution<double>(0, std);
  }
  void Shape(double mean, double std) {
    distribution_ = std::normal_distribution<double>(mean, std);
  }
  int GetInt() {
    return (int)round(distribution_(generator));
  }
  size_t GetSize() {
    return (size_t)round(distribution_(generator)); }
  double GetDouble() {
    return round(distribution_(generator)); }
};

class GammaDistribution : public Distribution {
 private:
  std::gamma_distribution<double> distribution_;
  //TODO: Is there a discrete gamma dist?
 public:
  GammaDistribution() = default;
  void Shape(double scale) {
    distribution_ = std::gamma_distribution<double>(1, scale);
  }
  void Shape(double shape, double scale) {
    distribution_ = std::gamma_distribution<double>(shape, scale);
  }
  int GetInt() {
    return (int)round(distribution_(generator));
  }
  size_t GetSize() {
    return (size_t)round(distribution_(generator));
  }
  double GetDouble() {
    return round(distribution_(generator));
  }
};

class ExponentialDistribution : public Distribution {
 private:
  std::exponential_distribution<double> distribution_;
  //TODO: add poisson dist
 public:
  ExponentialDistribution() = default;
  void Shape(double scale) {
    distribution_ = std::exponential_distribution<double>(scale);
  }
  int GetInt() { return (int)round(distribution_(generator)); }
  size_t GetSize() { return (size_t)round(distribution_(generator)); }
  double GetDouble() { return round(distribution_(generator)); }
};

class UniformDistribution : public Distribution {
 private:
  std::uniform_real_distribution<double> distribution_;
  //TODO: add int uniform dist
 public:
  UniformDistribution() = default;
  void Shape(size_t high) {
    distribution_ = std::uniform_real_distribution<double>(0, (double)high);
  }
  void Shape(double high) {
    distribution_ = std::uniform_real_distribution<double>(0, high);
  }
  void Shape(double low, double high) {
    distribution_ = std::uniform_real_distribution<double>(low, high);
  }
  int GetInt() {
    return (int)round(distribution_(generator));
  }
  size_t GetSize() {
    return (size_t)round(distribution_(generator));
  }
  double GetDouble() {
    return round(distribution_(generator));
  }
};

}  // namespace hshm

#endif  // HERMES_SHM_INCLUDE_HERMES_SHM_UTIL_RANDOM_H_
