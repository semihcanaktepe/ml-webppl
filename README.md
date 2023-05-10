# Machine Learning with WebPPL

## Matrix Operations
### Transpose a matrix

```
var M = Matrix([[1, 2, 3,], [4, 5, 6]])
var N = Matrix([[1,2], [3,4], [5,6]])
var O = Matrix([[1,2,3,4,5,6]])
```

```
var transpose = function(matrix){
  var dim1 = matrix.dims[0]
  var dim2 = matrix.dims[1]
  var indeces1 = _.range(dim1)
  var indeces2 = _.range(dim2)
  return Matrix(map(function(j){
    return map(function(i){return matrix.data[i*dim2+j]}, indeces1)}, indeces2))
}

//transpose(M)

T.transpose(M)
```

### Dot product of two vectors
```
var V1 = Vector([1, 2, 3, 8, 0])
var V2 = Vector([4, 5, 6, 7, 5])

var dot = function(vector1, vector2){
  var indeces = _.range(vector1.length)
  return vector1.length != vector2.length ?
    error("Vectors are not of the same length.") :
  sum(map(function(i){vector1.data[i]*vector2.data[i]}, indeces))
}

dot(V1, V2)
```

### Mathematical Operations on Matrices
```
var M = Matrix([[1, 2, 3], [4, 5, 6]])
var N = Matrix([[1, 2], [3, 4], [5, 6]])

var A = Matrix([[5, 0, 3], [7, 1, 6], [8, 2, 9]])
var B = Matrix([[3, 2, 4], [3, 4, 10], [5, 6, 12]])
```

#### Matrix multiplication (Matrix Product)
```
var mul = function(matrix1, matrix2){
  var M1dim0 = matrix1.dims[0]
  var M1dim1 = matrix1.dims[1]
  var M2dim0 = matrix2.dims[0]
  var M2dim1 = matrix2.dims[1]
  var M1indeces1 = _.range(M1dim0)
  var M2indeces1 = _.range(M2dim0)
  var M2indeces2 = _.range(M2dim1)
  return M1dim1 != M2dim0 ?
    error("Dimensions of matrices do not match.") : Matrix(map(function(i){
    return map(function(j){
      return sum(
        map(function(k){
          return matrix1.data[i*M1dim1+k]*matrix2.data[k*M2dim1+j]},
            M2indeces1))},
               M2indeces2)}, M1indeces1))
}
```
#### Matrix Addition
```
var add = function(matrix1, matrix2){
  var M1dim0 = matrix1.dims[0]
  var M1dim1 = matrix1.dims[1]
  var M2dim0 = matrix2.dims[0]
  var M2dim1 = matrix2.dims[1]
  var M1indeces1 = _.range(M1dim0)
  var M2indeces1 = _.range(M2dim0)
  var M2indeces2 = _.range(M2dim1)
  return (M1dim0 != M2dim0 && M1dim1 != M2dim1) ?
    error("Dimensions of matrices do not match.") : Matrix(map(function(i){
    map(function(j){
      return matrix1.data[i*M1dim0+j]+matrix2.data[i*M1dim0+j]},
        M1indeces1)}, M2indeces2))
}
```

#### Matrix Subtraction
```
var sub = function(matrix1, matrix2){
  var M1dim0 = matrix1.dims[0]
  var M1dim1 = matrix1.dims[1]
  var M2dim0 = matrix2.dims[0]
  var M2dim1 = matrix2.dims[1]
  var M1indeces1 = _.range(M1dim0)
  var M2indeces1 = _.range(M2dim0)
  var M2indeces2 = _.range(M2dim1)
  return (M1dim0 != M2dim0 && M1dim1 != M2dim1) ?
    error("Dimensions of matrices do not match.") : Matrix(map(function(i){
    map(function(j){
      return matrix1.data[i*M1dim0+j]-matrix2.data[i*M1dim0+j]},
        M1indeces1)}, M2indeces2))
}
```

#### Scalar multiplication of a Matrix
```
var scamul = function(matrix, scalar){
  var dim0 = matrix.dims[0]//2
  var dim1 = matrix.dims[1]//3
  var indeces0 = _.range(dim0)
  var indeces1 = _.range(dim1)
  return Matrix(map(function(i){
    map(function(j){
      matrix.data[i*dim0+j]*scalar}, indeces1)}, indeces0))
}

var X = Matrix([[1,0,1],[1,5,1],[1,15,2],[1,25,5],[1,35,11],[1,45,15],[1,55,34],[1,60,35]])
var Y = Matrix([[4],[5],[20],[14],[32],[22],[38],[43]])

var beta = mul(mul(T.inverse(mul(transpose(X), X)), transpose(X)), Y)
beta
```


### Random number generator based on a distribution

#### from k to j n number of evenly spaced numbers
```
var linspace = function(k,j,n){
  var step = (j-k)/(n-1)
  var f = function(x, arr){
    return arr.length == n-1 ?
      _.concat(arr, j) : f(x+step, _.concat(arr, x))
  }
  return f(k, [])
}

linspace(0,1,5)
```

```
var I = idMatrix(5)
var O = ones([5,5])
var Z = zeros([5,1])
var H = oneHot(3,5)
_.concat([1,2,3,4,5], 6)
```

## Gradient Descent Algorithm
```
var xs = _.range(-50, 50)
var ys = map(function(x){return 5+x*20}, xs)

var gradient_step = function(v, gradient, step_size){
  var step = map(function(i){return step_size*i}, gradient)
  return [v[0]+step[0], v[1]+step[1]]
}

var theta = [uniform(-1,1), uniform(-1,1)]

// Linear Gradient Function
var linear_gradient = function(x,y,theta){
  var intercept = theta[1]
  var slope = theta[0]
  var predicted = intercept + x*slope
  var error = predicted - y
  var squared_error = Math.pow(error, 2)
  var grad = [2*error*x, 2*error]
  return grad
}

var gradient_descent = function(epoch, learning_rate, theta){
  var values = map2(function(x,y){return linear_gradient(x,y,theta)}, xs, ys)
  var x_values = map(function(arr){return arr[0]}, values)
  var y_values = map(function(arr){return arr[1]}, values)
  var grad = [sum(x_values)/x_values.length, sum(y_values)/y_values.length]
  var updated_theta = gradient_step(theta, grad, -learning_rate)
  return epoch == 0 ?
    {theta:theta} : gradient_descent(epoch-1, learning_rate, updated_theta)
}

gradient_descent(5000, 0.001, theta)

```
## Stochastic Gradient Descent Algorithm

```
var stochastic_gradient_descent = function(epoch, learning_rate, theta){
  var xs = _.range(-50, 50)
  var ys = map(function(x){return 5+x*20}, xs)
  var u = map(function(i){
    var grad = linear_gradient(xs[i],ys[i],theta)
    var updated_theta = gradient_step(theta, grad, -learning_rate)
    return updated_theta
  }, _.range(100))
  return epoch == 0 ?
    {theta:theta} : stochastic_gradient_descent(epoch-1, learning_rate, u)
}

stochastic_gradient_descent(100, 0.001, theta)
```
