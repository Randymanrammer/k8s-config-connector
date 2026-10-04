// Copyright 2026 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// Package ai provides a small router for OpenAI-compatible chat completion
// gateways, with per-model retries, model fallback and JSON output parsing.
package ai

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"math/rand/v2"
	"net/http"
	"strconv"
	"strings"
	"time"
)

// ChatMessage is a single chat message.
type ChatMessage struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

// Policy describes which models to try for a task, in order.
type Policy struct {
	Models []string
	// MaxRetries is the number of retries per model for retryable errors.
	MaxRetries int
	// Backoff is the base delay between retries (doubled each retry).
	Backoff time.Duration
	// MaxBackoff caps the delay between retries. Defaults to 30s.
	MaxBackoff time.Duration
}

const (
	defaultBackoff    = 500 * time.Millisecond
	defaultMaxBackoff = 30 * time.Second
	maxRetriesLimit   = 10
	maxErrorBodyBytes = 2048
	maxResponseBytes  = 16 << 20
)

// Usage reports token usage.
type Usage struct {
	PromptTokens     int `json:"prompt_tokens"`
	CompletionTokens int `json:"completion_tokens"`
	TotalTokens      int `json:"total_tokens"`
}

// CallRecord is passed to the telemetry hook after every attempt.
type CallRecord struct {
	Task     string
	Model    string
	Attempt  int
	Duration time.Duration
	Usage    Usage
	Err      error
}

// Result is a successful run.
type Result[T any] struct {
	Data    T
	Model   string
	Usage   Usage
	Attempt int
}

// HTTPError is a non-2xx response from the gateway.
type HTTPError struct {
	StatusCode int
	Body       string
	// RetryAfter is the server-requested delay, if any.
	RetryAfter time.Duration
}

func (e *HTTPError) Error() string {
	return fmt.Sprintf("gateway returned status %d: %s", e.StatusCode, e.Body)
}

// ErrEmptyResponse is returned when the model returns no content.
var ErrEmptyResponse = errors.New("empty model response")

// ErrSchema is returned when the response cannot be decoded into the target type.
var ErrSchema = errors.New("response does not match schema")

// AllModelsFailedError is returned when every candidate model failed.
type AllModelsFailedError struct {
	Task     string
	Failures []error
}

func (e *AllModelsFailedError) Error() string {
	return fmt.Sprintf("all models failed for task %q: %v", e.Task, errors.Join(e.Failures...))
}

func (e *AllModelsFailedError) Unwrap() []error { return e.Failures }

// Router calls an OpenAI-compatible chat completions endpoint.
type Router struct {
	BaseURL    string
	APIKey     string
	HTTPClient *http.Client
	Headers    map[string]string
	// Telemetry, if set, is called after every attempt.
	Telemetry func(CallRecord)
	// Logger receives debug logs for each attempt. Defaults to a discard logger.
	Logger *slog.Logger
	// Validate, if set, is called on decoded output; a non-nil error is
	// treated as a schema failure.
	Validate func(any) error
	// Sleep is overridable for tests.
	Sleep func(ctx context.Context, d time.Duration) error
}

// NewRouter returns a Router with defaults.
func NewRouter(baseURL, apiKey string) *Router {
	return &Router{
		BaseURL:    strings.TrimRight(baseURL, "/"),
		APIKey:     apiKey,
		HTTPClient: &http.Client{Timeout: 2 * time.Minute},
	}
}

type chatRequest struct {
	Model          string         `json:"model"`
	Messages       []ChatMessage  `json:"messages"`
	ResponseFormat map[string]any `json:"response_format,omitempty"`
}

type chatResponse struct {
	Choices []struct {
		Message struct {
			Content string `json:"content"`
		} `json:"message"`
	} `json:"choices"`
	Usage Usage `json:"usage"`
}

// RunModelTask runs the task against each model in the policy until one
// returns output that decodes into T. Retries happen here (not in the HTTP
// client) so they cannot multiply with fallbacks.
func RunModelTask[T any](ctx context.Context, r *Router, task string, policy Policy, messages []ChatMessage) (*Result[T], error) {
	if r == nil || r.BaseURL == "" {
		return nil, errors.New("router base URL is not configured")
	}
	if len(policy.Models) == 0 {
		return nil, fmt.Errorf("no models configured for task %q", task)
	}
	if len(messages) == 0 {
		return nil, fmt.Errorf("no messages provided for task %q", task)
	}
	policy = policy.normalized()
	log := r.logger()
	var failures []error
	for _, model := range policy.Models {
		for attempt := 0; attempt <= policy.MaxRetries; attempt++ {
			if err := ctx.Err(); err != nil {
				return nil, err
			}
			start := time.Now()
			data, usage, err := callOnce[T](ctx, r, model, messages)
			if r.Telemetry != nil {
				r.Telemetry(CallRecord{Task: task, Model: model, Attempt: attempt, Duration: time.Since(start), Usage: usage, Err: err})
			}
			if err == nil {
				return &Result[T]{Data: data, Model: model, Usage: usage, Attempt: attempt}, nil
			}
			log.DebugContext(ctx, "model attempt failed", "task", task, "model", model, "attempt", attempt, "duration", time.Since(start), "error", err)
			failures = append(failures, fmt.Errorf("%s (attempt %d): %w", model, attempt, err))
			if !retryable(err) {
				if isFatal(err) {
					return nil, err
				}
				break
			}
			if attempt < policy.MaxRetries {
				if err := r.sleep(ctx, policy.delay(attempt, err)); err != nil {
					return nil, err
				}
			}
		}
	}
	return nil, &AllModelsFailedError{Task: task, Failures: failures}
}

func callOnce[T any](ctx context.Context, r *Router, model string, messages []ChatMessage) (T, Usage, error) {
	var zero T
	body, err := json.Marshal(chatRequest{
		Model:          model,
		Messages:       messages,
		ResponseFormat: map[string]any{"type": "json_object"},
	})
	if err != nil {
		return zero, Usage{}, err
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, r.BaseURL+"/chat/completions", bytes.NewReader(body))
	if err != nil {
		return zero, Usage{}, err
	}
	req.Header.Set("Content-Type", "application/json")
	if r.APIKey != "" {
		req.Header.Set("Authorization", "Bearer "+r.APIKey)
	}
	for k, v := range r.Headers {
		req.Header.Set(k, v)
	}
	client := r.HTTPClient
	if client == nil {
		client = http.DefaultClient
	}
	resp, err := client.Do(req)
	if err != nil {
		return zero, Usage{}, err
	}
	defer resp.Body.Close()
	respBody, err := io.ReadAll(io.LimitReader(resp.Body, maxResponseBytes))
	if err != nil {
		return zero, Usage{}, err
	}
	if resp.StatusCode < 200 || resp.StatusCode > 299 {
		msg := string(respBody)
		if len(msg) > maxErrorBodyBytes {
			msg = msg[:maxErrorBodyBytes] + "...(truncated)"
		}
		return zero, Usage{}, &HTTPError{StatusCode: resp.StatusCode, Body: msg, RetryAfter: parseRetryAfter(resp.Header.Get("Retry-After"))}
	}
	var cr chatResponse
	if err := json.Unmarshal(respBody, &cr); err != nil {
		return zero, Usage{}, fmt.Errorf("decoding gateway response: %w", err)
	}
	if len(cr.Choices) == 0 || strings.TrimSpace(cr.Choices[0].Message.Content) == "" {
		return zero, cr.Usage, ErrEmptyResponse
	}
	var out T
	if err := json.Unmarshal([]byte(extractJSON(cr.Choices[0].Message.Content)), &out); err != nil {
		return zero, cr.Usage, fmt.Errorf("%w: %v", ErrSchema, err)
	}
	if r.Validate != nil {
		if err := r.Validate(out); err != nil {
			return zero, cr.Usage, fmt.Errorf("%w: %v", ErrSchema, err)
		}
	}
	return out, cr.Usage, nil
}

// extractJSON strips markdown code fences that models sometimes add.
func extractJSON(s string) string {
	s = strings.TrimSpace(s)
	if strings.HasPrefix(s, "```") {
		s = strings.TrimPrefix(s, "```json")
		s = strings.TrimPrefix(s, "```")
		s = strings.TrimSuffix(strings.TrimSpace(s), "```")
	}
	return strings.TrimSpace(s)
}

// isFatal reports errors that no other model would fix (bad credentials).
func isFatal(err error) bool {
	var he *HTTPError
	return errors.As(err, &he) && (he.StatusCode == http.StatusUnauthorized || he.StatusCode == http.StatusForbidden)
}

// retryable reports whether retrying the same model may help.
func retryable(err error) bool {
	var he *HTTPError
	if errors.As(err, &he) {
		return he.StatusCode == http.StatusTooManyRequests || he.StatusCode >= 500
	}
	if errors.Is(err, ErrEmptyResponse) || errors.Is(err, ErrSchema) || errors.Is(err, context.Canceled) || errors.Is(err, context.DeadlineExceeded) {
		return false
	}
	// Network errors.
	return true
}

func (r *Router) sleep(ctx context.Context, d time.Duration) error {
	if r.Sleep != nil {
		return r.Sleep(ctx, d)
	}
	t := time.NewTimer(d)
	defer t.Stop()
	select {
	case <-ctx.Done():
		return ctx.Err()
	case <-t.C:
		return nil
	}
}

func (p Policy) normalized() Policy {
	if p.MaxRetries < 0 {
		p.MaxRetries = 0
	}
	if p.MaxRetries > maxRetriesLimit {
		p.MaxRetries = maxRetriesLimit
	}
	if p.Backoff <= 0 {
		p.Backoff = defaultBackoff
	}
	if p.MaxBackoff <= 0 {
		p.MaxBackoff = defaultMaxBackoff
	}
	return p
}

// delay returns the wait before the next retry: exponential backoff with
// jitter, capped at MaxBackoff, honoring Retry-After when larger.
func (p Policy) delay(attempt int, err error) time.Duration {
	d := p.MaxBackoff
	if attempt < 31 {
		if b := p.Backoff << attempt; b > 0 && b < d {
			d = b
		}
	}
	// Full jitter in [d/2, d].
	d = d/2 + time.Duration(rand.Int64N(int64(d/2)+1))
	var he *HTTPError
	if errors.As(err, &he) && he.RetryAfter > d {
		d = he.RetryAfter
		if d > p.MaxBackoff {
			d = p.MaxBackoff
		}
	}
	return d
}

func parseRetryAfter(v string) time.Duration {
	if secs, err := strconv.Atoi(strings.TrimSpace(v)); err == nil && secs > 0 {
		return time.Duration(secs) * time.Second
	}
	return 0
}

func (r *Router) logger() *slog.Logger {
	if r.Logger != nil {
		return r.Logger
	}
	return slog.New(slog.NewTextHandler(io.Discard, nil))
}
