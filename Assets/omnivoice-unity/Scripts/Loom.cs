using UnityEngine;
using System.Collections.Generic;
using System;
using System.Threading;
using Debug = UnityEngine.Debug;

/// <summary>
/// 多线程调度器 — 在 Unity 主线程上执行回调，同时提供后台线程能力。
/// 
/// 核心用法：
///   1. Loom.Initialize()          — 初始化（线程安全，可多次调用）
///   2. Loom.QueueOnMainThread(...) — 将回调排入主线程 Update 执行
///   3. Loom.QueueOnMainThread(..., time) — 延迟执行
///   4. Loom.RunAsync(...)          — 在后台线程池执行任务
/// </summary>
public class Loom : MonoBehaviour
{
    public static int maxThreads = 8;
    static int numThreads;

    private static Loom _current;
    private static readonly object _initLock = new object();
    static bool initialized;

    /// <summary>
    /// 获取 Loom 实例，首次访问时自动初始化。
    /// </summary>
    public static Loom Current
    {
        get
        {
            Initialize();
            return _current;
        }
    }

    /// <summary>
    /// 初始化 Loom，在场景中创建不会被销毁的 GameObject。
    /// 线程安全，可多次调用。
    /// </summary>
    public static void Initialize()
    {
        if (initialized) return;

        lock (_initLock)
        {
            if (initialized) return;
            initialized = true;

            var g = new GameObject("Loom");
            DontDestroyOnLoad(g);
            _current = g.AddComponent<Loom>();
            Debug.Log("[Loom] 初始化完成");
        }
    }

    /// <summary>
    /// 检查 Loom 是否已初始化（不触发自动初始化）。
    /// </summary>
    public static bool IsInitialized()
    {
        return initialized && _current != null;
    }

    // ─────────────────────────────────────────────────────────────
    // 主线程回调队列
    // ─────────────────────────────────────────────────────────────

    private List<Action> _actions = new List<Action>();

    public struct DelayedQueueItem
    {
        public float time;
        public Action action;
    }

    private List<DelayedQueueItem> _delayed = new List<DelayedQueueItem>();

    // 每帧复用的工作缓冲区（避免 Update 中的 GC 分配）
    private List<Action> _currentActions = new List<Action>();
    private List<DelayedQueueItem> _currentDelayed = new List<DelayedQueueItem>();

    /// <summary>
    /// 将回调排入主线程，下一帧 Update 时执行。
    /// </summary>
    public static void QueueOnMainThread(Action action)
    {
        QueueOnMainThread(action, 0f);
    }

    /// <summary>
    /// 将回调延迟到指定时间后执行。
    /// </summary>
    public static void QueueOnMainThread(Action action, float delaySeconds)
    {
        var loom = Current;
        if (loom == null) return;

        if (delaySeconds > 0f)
        {
            lock (loom._delayed)
            {
                loom._delayed.Add(new DelayedQueueItem { time = Time.time + delaySeconds, action = action });
            }
        }
        else
        {
            lock (loom._actions)
            {
                loom._actions.Add(action);
            }
        }
    }

    // ─────────────────────────────────────────────────────────────
    // 后台线程
    // ─────────────────────────────────────────────────────────────

    /// <summary>
    /// 在线程池上异步执行任务。
    /// 如果当前线程数已达上限 maxThreads，会等待有空闲槽位再执行。
    /// </summary>
    public static void RunAsync(Action action)
    {
        Initialize();

        // 等待空闲线程槽
        while (numThreads >= maxThreads)
        {
            Thread.Sleep(1);
        }

        Interlocked.Increment(ref numThreads);
        ThreadPool.QueueUserWorkItem(RunAction, action);
    }

    private static void RunAction(object state)
    {
        try
        {
            ((Action)state)();
        }
        catch (Exception ex)
        {
            Debug.LogError($"[Loom] 后台任务异常: {ex.Message}\n{ex.StackTrace}");
        }
        finally
        {
            Interlocked.Decrement(ref numThreads);
        }
    }

    // ─────────────────────────────────────────────────────────────
    // Unity 生命周期
    // ─────────────────────────────────────────────────────────────

    void OnDisable()
    {
        if (_current == this)
        {
            _current = null;
        }
    }

    /// <summary>
    /// 每帧处理主线程回调和延迟回调。
    /// 使用复用缓冲区，零 GC 分配。
    /// </summary>
    void Update()
    {
        // ── 处理主线程回调 ──
        lock (_actions)
        {
            _currentActions.Clear();
            _currentActions.AddRange(_actions);
            _actions.Clear();
        }

        for (int i = 0; i < _currentActions.Count; i++)
        {
            _currentActions[i]?.Invoke();
        }

        // ── 处理延迟回调 ──
        lock (_delayed)
        {
            _currentDelayed.Clear();
            float now = Time.time;

            // 遍历查找到期项（避免 LINQ Where 的 GC 分配）
            for (int i = 0; i < _delayed.Count; i++)
            {
                if (_delayed[i].time <= now)
                {
                    _currentDelayed.Add(_delayed[i]);
                }
            }

            // 移除已到期的项（从后往前移除，避免索引错乱）
            for (int i = _currentDelayed.Count - 1; i >= 0; i--)
            {
                _delayed.Remove(_currentDelayed[i]);
            }
        }

        for (int i = 0; i < _currentDelayed.Count; i++)
        {
            _currentDelayed[i].action?.Invoke();
        }
    }
}
