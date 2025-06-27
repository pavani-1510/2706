**📘 Trilogy Interview Revision Sheet**

---

### 1. Maximum Subarray Sum (Leetcode 53)

**Problem**: Find the largest sum of any contiguous subarray.

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

**Problem**: Return the sum of three integers closest to the target.

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

**Problem**: Count number of connected '1's islands.

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

**Problem**: Count ways to make amount with infinite coins.

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

**Problem**: Can we divide nodes into 2 sets with no same-set edges?

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

**Problem**: Sort linked list in O(n log n) time.

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

**Problem**: Max profit with 1 buy and 1 sell.

**Approach**: Track min price

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

**Problem**: Count all substrings that are palindromes.

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

**Problem**: Insert minimum chars to make `s` a palindrome.

**Approach**: LCS(s, reverse(s)) → answer = len - LCS

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

**Problem**: Split array into `k` parts to minimize largest sum.

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

**Problem**: Count number of subarrays with sum = k

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

**Problem**: Can we finish all courses (detect cycles)?

**Approach**: Topological Sort (Kahn's Algo)

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
