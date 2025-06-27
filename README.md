**📘 Trilogy Interview Revision Sheet with Problem Statements**

---

### 1. Maximum Subarray Sum (Leetcode 53)

**Problem**: Given an integer array `nums`, find the contiguous subarray (containing at least one number) which has the largest sum and return its sum.

**Approach**: Kadane's Algorithm

```java
public int maxSubArray(int[] nums) {
    int max = nums[0], curr = nums[0];
    for (int i = 1; i < nums.length; i++) {
        curr = Math.max(nums[i], curr + nums[i]);
        max = Math.max(max, curr);
    }
    return max;
}
```

---

### 2. 3Sum Closest (Leetcode 16)

**Problem**: Given an integer array `nums` of length `n` and an integer `target`, find three integers in `nums` such that the sum is closest to `target`. Return the sum of the three integers.

**Approach**: Sort + Two Pointers

```java
public int threeSumClosest(int[] nums, int target) {
    Arrays.sort(nums);
    int closest = nums[0] + nums[1] + nums[2];
    for (int i = 0; i < nums.length - 2; i++) {
        int left = i + 1, right = nums.length - 1;
        while (left < right) {
            int sum = nums[i] + nums[left] + nums[right];
            if (Math.abs(target - sum) < Math.abs(target - closest)) {
                closest = sum;
            }
            if (sum < target) left++;
            else right--;
        }
    }
    return closest;
}
```

---

### 3. Number of Islands (Leetcode 200)

**Problem**: Given a 2D grid map of `'1'`s (land) and `'0'`s (water), count the number of islands. An island is surrounded by water and is formed by connecting adjacent lands horizontally or vertically.

**Approach**: DFS

```java
public int numIslands(char[][] grid) {
    int count = 0;
    for (int i = 0; i < grid.length; i++) {
        for (int j = 0; j < grid[0].length; j++) {
            if (grid[i][j] == '1') {
                dfs(grid, i, j);
                count++;
            }
        }
    }
    return count;
}

private void dfs(char[][] grid, int i, int j) {
    if (i < 0 || j < 0 || i >= grid.length || j >= grid[0].length || grid[i][j] == '0') return;
    grid[i][j] = '0';
    dfs(grid, i + 1, j);
    dfs(grid, i - 1, j);
    dfs(grid, i, j + 1);
    dfs(grid, i, j - 1);
}
```

---

### 4. Coin Change II (Leetcode 518)

**Problem**: You are given coins of different denominations and a total amount. Return the number of combinations that make up that amount.

**Approach**: 1D Bottom-Up DP

```java
public int change(int amount, int[] coins) {
    int[] dp = new int[amount + 1];
    dp[0] = 1;
    for (int coin : coins) {
        for (int i = coin; i <= amount; i++) {
            dp[i] += dp[i - coin];
        }
    }
    return dp[amount];
}
```

---

### 5. Is Graph Bipartite (Leetcode 785)

**Problem**: Given an undirected graph, return true if the graph is bipartite. A graph is bipartite if its vertices can be divided into two disjoint sets such that every edge connects a vertex from one set to the other.

**Approach**: BFS Coloring

```java
public boolean isBipartite(int[][] graph) {
    int n = graph.length;
    int[] color = new int[n];
    Arrays.fill(color, -1);

    for (int i = 0; i < n; i++) {
        if (color[i] == -1) {
            Queue<Integer> q = new LinkedList<>();
            q.offer(i);
            color[i] = 0;
            while (!q.isEmpty()) {
                int node = q.poll();
                for (int neighbor : graph[node]) {
                    if (color[neighbor] == color[node]) return false;
                    if (color[neighbor] == -1) {
                        color[neighbor] = 1 - color[node];
                        q.offer(neighbor);
                    }
                }
            }
        }
    }
    return true;
}
```

---

### 6. Merge Sort Linked List (Leetcode 148)

**Problem**: Sort a linked list in O(n log n) time and constant space complexity.

**Approach**: Merge Sort on List

```java
public ListNode sortList(ListNode head) {
    if (head == null || head.next == null) return head;
    ListNode mid = getMid(head);
    ListNode left = sortList(head);
    ListNode right = sortList(mid);
    return merge(left, right);
}

private ListNode getMid(ListNode head) {
    ListNode slow = head, fast = head.next;
    while (fast != null && fast.next != null) {
        slow = slow.next;
        fast = fast.next.next;
    }
    ListNode mid = slow.next;
    slow.next = null;
    return mid;
}

private ListNode merge(ListNode l1, ListNode l2) {
    ListNode dummy = new ListNode(), tail = dummy;
    while (l1 != null && l2 != null) {
        if (l1.val < l2.val) {
            tail.next = l1; l1 = l1.next;
        } else {
            tail.next = l2; l2 = l2.next;
        }
        tail = tail.next;
    }
    tail.next = (l1 != null) ? l1 : l2;
    return dummy.next;
}
```

---

### 7. Best Time to Buy and Sell Stock (Leetcode 121)

**Problem**: Find the maximum profit from buying and selling one stock.

**Approach**: Track min price and max profit

```java
public int maxProfit(int[] prices) {
    int minPrice = Integer.MAX_VALUE, maxProfit = 0;
    for (int price : prices) {
        if (price < minPrice) minPrice = price;
        else maxProfit = Math.max(maxProfit, price - minPrice);
    }
    return maxProfit;
}
```

---

### 8. Palindromic Substrings (Leetcode 647)

**Problem**: Return the number of palindromic substrings in the input string.

**Approach**: Expand Around Center

```java
public int countSubstrings(String s) {
    int count = 0;
    for (int i = 0; i < s.length(); i++) {
        count += expand(s, i, i);
        count += expand(s, i, i + 1);
    }
    return count;
}

private int expand(String s, int left, int right) {
    int count = 0;
    while (left >= 0 && right < s.length() && s.charAt(left) == s.charAt(right)) {
        count++; left--; right++;
    }
    return count;
}
```

---

### 9. Minimum Insertions to Make Palindrome (Leetcode 1312)

**Problem**: Find the minimum number of insertions to make the string a palindrome.

**Approach**: Use LCS(s, reverse(s)); answer = len - LCS

```java
public int minInsertions(String s) {
    String rev = new StringBuilder(s).reverse().toString();
    int n = s.length();
    int[][] dp = new int[n + 1][n + 1];
    for (int i = 1; i <= n; i++) {
        for (int j = 1; j <= n; j++) {
            if (s.charAt(i - 1) == rev.charAt(j - 1))
                dp[i][j] = 1 + dp[i - 1][j - 1];
            else
                dp[i][j] = Math.max(dp[i - 1][j], dp[i][j - 1]);
        }
    }
    return n - dp[n][n];
}
```

---

### 10. Split Array Largest Sum (Leetcode 410)

**Problem**: Split array into `k` parts to minimize the largest subarray sum.

**Approach**: Binary Search + Greedy Check

```java
public int splitArray(int[] nums, int k) {
    int left = 0, right = 0;
    for (int num : nums) {
        left = Math.max(left, num);
        right += num;
    }
    while (left < right) {
        int mid = (left + right) / 2;
        if (canSplit(nums, k, mid)) right = mid;
        else left = mid + 1;
    }
    return left;
}

private boolean canSplit(int[] nums, int k, int maxSum) {
    int count = 1, sum = 0;
    for (int num : nums) {
        if (sum + num > maxSum) {
            count++;
            sum = num;
        } else sum += num;
    }
    return count <= k;
}
```

---

### 11. Subarray Sum Equals K (Leetcode 560)

**Problem**: Find the number of subarrays whose sum equals `k`.

**Approach**: Prefix Sum + HashMap

```java
public int subarraySum(int[] nums, int k) {
    Map<Integer, Integer> map = new HashMap<>();
    map.put(0, 1);
    int sum = 0, count = 0;
    for (int num : nums) {
        sum += num;
        count += map.getOrDefault(sum - k, 0);
        map.put(sum, map.getOrDefault(sum, 0) + 1);
    }
    return count;
}
```

---

### 12. Course Schedule (Leetcode 207)

**Problem**: Given `numCourses` and a list of prerequisites, determine if it's possible to finish all courses (i.e., no cyclic dependencies).

**Approach**: Topological Sort (Kahn's Algorithm)

```java
public boolean canFinish(int numCourses, int[][] prerequisites) {
    List<Integer>[] graph = new ArrayList[numCourses];
    int[] inDegree = new int[numCourses];
    for (int i = 0; i < numCourses; i++) graph[i] = new ArrayList<>();
    for (int[] p : prerequisites) {
        graph[p[1]].add(p[0]);
        inDegree[p[0]]++;
    }
    Queue<Integer> q = new LinkedList<>();
    for (int i = 0; i < numCourses; i++) if (inDegree[i] == 0) q.add(i);
    int finished = 0;
    while (!q.isEmpty()) {
        int course = q.poll();
        finished++;
        for (int next : graph[course]) {
            inDegree[next]--;
            if (inDegree[next] == 0) q.add(next);
        }
    }
    return finished == numCourses;
}
```

### **13. Diameter of Binary Tree – LeetCode 543**

#### ✅ Problem Statement:

Given the root of a binary tree, return the **length of the diameter** of the tree.
The diameter is the **longest path between any two nodes**, counted by **number of edges**, not nodes.



#### 🧠 Approach: **post-order DFS**.


#### 💻 Code:

```java
class Solution {
    int maxDiameter = 0;

    public int diameterOfBinaryTree(TreeNode root) {
        dfs(root);
        return maxDiameter;
    }

    private int dfs(TreeNode node) {
        if (node == null) return 0;

        int left = dfs(node.left);
        int right = dfs(node.right);
        maxDiameter = Math.max(maxDiameter, left + right);

        return 1 + Math.max(left, right);
    }
}
```

---

### **14. Longest Substring Without Repeating Characters – LeetCode 3**

#### ✅ Problem Statement:

Given a string `s`, return the length of the **longest substring without repeating characters**.


#### 🧠 Approach: **sliding window** with `HashSet` or `Map`.



#### 💻 Code:

```java
class Solution {
    public int lengthOfLongestSubstring(String s) {
        Set<Character> set = new HashSet<>();
        int left = 0, maxLen = 0;

        for (int right = 0; right < s.length(); right++) {
            while (set.contains(s.charAt(right))) {
                set.remove(s.charAt(left++));
            }
            set.add(s.charAt(right));
            maxLen = Math.max(maxLen, right - left + 1);
        }

        return maxLen;
    }
}
```

---

### **15. Top K Frequent Elements – LeetCode 347**

#### ✅ Problem Statement:

Given an integer array `nums` and an integer `k`, return the `k` most frequent elements.


#### 🧠 Bucket Sort Approach:



#### 💻 Code:

```java
class Solution {
    public int[] topKFrequent(int[] nums, int k) {
        Map<Integer, Integer> freqMap = new HashMap<>();
        for (int num : nums)
            freqMap.put(num, freqMap.getOrDefault(num, 0) + 1);

        List<Integer>[] bucket = new List[nums.length + 1];
        for (int key : freqMap.keySet()) {
            int freq = freqMap.get(key);
            if (bucket[freq] == null)
                bucket[freq] = new ArrayList<>();
            bucket[freq].add(key);
        }

        List<Integer> result = new ArrayList<>();
        for (int i = bucket.length - 1; i >= 0 && result.size() < k; i--) {
            if (bucket[i] != null)
                result.addAll(bucket[i]);
        }

        int[] ans = new int[k];
        for (int i = 0; i < k; i++)
            ans[i] = result.get(i);

        return ans;
    }
}
```

---



## ✅ **16. Minimum Depth of Binary Tree - LeetCode 111**

---

### 🎯 **Problem Statement Recap**:

Given a binary tree, return the **minimum depth**, i.e., the shortest path from the **root** to the **nearest leaf node**.

* A **leaf** is a node with **no left or right child**.


## ✅ 1. DFS (Depth-First Search) Approach

### 💻 Java Code:

```java
class Solution {
    public int minDepth(TreeNode root) {
        if (root == null) return 0;

        // If one of the subtrees is null, return the depth of the other + 1
        if (root.left == null) return 1 + minDepth(root.right);
        if (root.right == null) return 1 + minDepth(root.left);

        // If both exist, take the minimum
        return 1 + Math.min(minDepth(root.left), minDepth(root.right));
    }
}
```



## ✅ 2. BFS (Breadth-First Search) Approach


### 💻 Java Code:

```java
class Solution {
    public int minDepth(TreeNode root) {
        if (root == null) return 0;

        Queue<TreeNode> queue = new LinkedList<>();
        queue.offer(root);
        int depth = 1;

        while (!queue.isEmpty()) {
            int size = queue.size();

            for (int i = 0; i < size; i++) {
                TreeNode node = queue.poll();

                // Leaf node
                if (node.left == null && node.right == null)
                    return depth;

                if (node.left != null) queue.offer(node.left);
                if (node.right != null) queue.offer(node.right);
            }
            depth++;
        }

        return depth;
    }
}
```

---

### **17. Break Number (Integer Break) – LeetCode 343**

#### ✅ Problem Statement:

Given an integer `n`, break it into the sum of **at least two** positive integers, and return the **maximum product** you can get.


#### 🧠 Greedy + Math Approach:


#### 💻 Code:

```java
class Solution {
    public int integerBreak(int n) {
        if (n == 2) return 1;
        if (n == 3) return 2;

        int product = 1;
        while (n > 4) {
            product *= 3;
            n -= 3;
        }
        return product * n;
    }
}
```
---

## ✅ Problem 

You are given:

* An **undirected**, **weighted**, **connected** graph with `V` vertices (numbered from `0` to `V-1`) and `E` edges.
* The graph is described using a 2D array `edges[][]`, where `edges[i] = [u, v, w]` represents an edge between vertices `u` and `v` with weight `w`.
* A source vertex `src`.

You need to **find the shortest distance from the source vertex `src` to every other vertex** in the graph and return it as an array.

## ✅ Approach:  **Dijkstra’s Algorithm** with a **Min-Heap (PriorityQueue)**:


```java
import java.util.*;

class Solution {
    static class Pair {
        int node, dist;
        Pair(int node, int dist) {
            this.node = node;
            this.dist = dist;
        }
    }

    public static int[] dijkstra(int V, int[][] edges, int src) {
        List<List<Pair>> graph = new ArrayList<>();
        for (int i = 0; i < V; i++) graph.add(new ArrayList<>());
        for (int[] edge : edges) {
            int u = edge[0], v = edge[1], w = edge[2];
            graph.get(u).add(new Pair(v, w));
            graph.get(v).add(new Pair(u, w));
        }

        int[] dist = new int[V];
        Arrays.fill(dist, Integer.MAX_VALUE);
        dist[src] = 0;

        PriorityQueue<Pair> pq = new PriorityQueue<>(Comparator.comparingInt(a -> a.dist));
        pq.add(new Pair(src, 0));

        while (!pq.isEmpty()) {
            Pair current = pq.poll();
            int u = current.node;
            int d = current.dist;

            for (Pair neighbor : graph.get(u)) {
                int v = neighbor.node, w = neighbor.dist;
                if (d + w < dist[v]) {
                    dist[v] = d + w;
                    pq.add(new Pair(v, dist[v]));
                }
            }
        }

        return dist;
    }

    public static void main(String[] args) {
        int V = 5;
        int[][] edges = {
            {0, 1, 4}, {0, 2, 8}, {1, 4, 6}, {2, 3, 2}, {3, 4, 10}
        };
        int src = 0;

        int[] result = dijkstra(V, edges, src);
        System.out.println(Arrays.toString(result));  // Output: [0, 4, 8, 10, 10]
    }
}
```

---


### 🧾 **Problem :**

You are given an integer array `arr[]` and an integer `k`. Your task is to **count the number of subarrays** whose elements’ XOR is equal to `k`.

### ✅ **Approach :** Optimal - Prefix XOR + HashMap


```java
import java.util.*;

public class CountSubarraysWithXOR {
    public static int countSubarrays(int[] arr, int k) {
        Map<Integer, Integer> map = new HashMap<>();
        int xor = 0, count = 0;

        for (int num : arr) {
            xor ^= num;
            if (xor == k) count++;
            int required = xor ^ k;
            count += map.getOrDefault(required, 0);
            map.put(xor, map.getOrDefault(xor, 0) + 1);
        }

        return count;
    }

    public static void main(String[] args) {
        System.out.println(countSubarrays(new int[]{4, 2, 2, 6, 4}, 6)); // Output: 4
        System.out.println(countSubarrays(new int[]{5, 6, 7, 8, 9}, 5)); // Output: 2
        System.out.println(countSubarrays(new int[]{1, 1, 1, 1}, 0));    // Output: 4
    }
}
```

---


## ✅ Problem :

You are given two arrays:

* `deadline[]`: deadlines for each job
* `profit[]`: profit for each job

Each job takes **1 unit of time** to complete and must be finished **on or before its deadline** to earn profit. You can schedule **only one job at a time**.

Your task is to find:

* The **maximum number of jobs** that can be completed within their deadlines.
* The **maximum profit** you can earn from those jobs.



## ✅ Approach: (Greedy + Sorting)



```java
import java.util.*;

class Job {
    int deadline, profit;
    Job(int d, int p) {
        deadline = d;
        profit = p;
    }
}

public class JobSequencing {

    public static int[] jobScheduling(int[] deadline, int[] profit) {
        int n = deadline.length;
        Job[] jobs = new Job[n];

        for (int i = 0; i < n; i++) {
            jobs[i] = new Job(deadline[i], profit[i]);
        }

        Arrays.sort(jobs, (a, b) -> b.profit - a.profit);

        int maxDeadline = 0;
        for (Job job : jobs) {
            maxDeadline = Math.max(maxDeadline, job.deadline);
        }

        boolean[] slot = new boolean[maxDeadline + 1];
        int jobCount = 0, maxProfit = 0;

        for (Job job : jobs) {
            for (int t = job.deadline; t > 0; t--) {
                if (!slot[t]) {
                    slot[t] = true;
                    jobCount++;
                    maxProfit += job.profit;
                    break;
                }
            }
        }

        return new int[]{jobCount, maxProfit};
    }

    public static void main(String[] args) {
        System.out.println(Arrays.toString(jobScheduling(new int[]{4, 1, 1, 1}, new int[]{20, 10, 40, 30})));  // [2, 60]
        System.out.println(Arrays.toString(jobScheduling(new int[]{2, 1, 2, 1, 1}, new int[]{100, 19, 27, 25, 15})));  // [2, 127]
        System.out.println(Arrays.toString(jobScheduling(new int[]{3, 1, 2, 2}, new int[]{50, 10, 20, 30})));  // [3, 100]
    }
}
```

