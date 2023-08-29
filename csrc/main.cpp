#include <iostream>
#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>


int main() {
    const int d = 4;
    const int N = d * d;
    float *test = new float[N];
    for (int i = 0; i < N; ++i) {
        test[i] = i;
    }

    glm::mat4 A = glm::make_mat4(test);
    for (int i = 0; i < d; ++i) {
        for (int j = 0; j < d; ++j) {
            std::cout << A[i][j] << std::endl;
        }
    }

    glm::mat3 B = glm::mat3(glm::vec3(A[0]),glm::vec3(A[1]), glm::vec3(A[2]));
    for (int i = 0; i < d-1; ++i) {
        for (int j = 0; j < d-1; ++j) {
            std::cout << B[i][j] << std::endl;
        }
    }

    delete[] test;
    return 0;
}
