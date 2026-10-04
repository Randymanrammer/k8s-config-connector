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

package ai

import (
	"context"
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"
)

type out struct {
	Name string `json:"name"`
}

func newTestRouter(t *testing.T, h http.HandlerFunc) *Router {
	srv := httptest.NewServer(h)
	t.Cleanup(srv.Close)
	r := NewRouter(srv.URL, "key")
	r.Sleep = func(context.Context, time.Duration) error { return nil }
	return r
}

func reply(w http.ResponseWriter, content string) {
	json.NewEncoder(w).Encode(map[string]any{
		"choices": []any{map[string]any{"message": map[string]any{"content": content}}},
		"usage":   map[string]any{"total_tokens": 3},
	})
}

func TestFallbackAndRetry(t *testing.T) {
	var calls []string
	r := newTestRouter(t, func(w http.ResponseWriter, req *http.Request) {
		var body chatRequest
		json.NewDecoder(req.Body).Decode(&body)
		calls = append(calls, body.Model)
		if body.Model == "a" {
			w.WriteHeader(http.StatusServiceUnavailable)
			return
		}
		reply(w, "```json\n{\"name\":\"x\"}\n```")
	})
	res, err := RunModelTask[out](context.Background(), r, "t", Policy{Models: []string{"a", "b"}, MaxRetries: 1}, nil)
	if err != nil {
		t.Fatal(err)
	}
	if res.Data.Name != "x" || res.Model != "b" {
		t.Fatalf("unexpected result %+v", res)
	}
	if len(calls) != 3 {
		t.Fatalf("calls = %v, want a,a,b", calls)
	}
}

func TestAllFailed(t *testing.T) {
	r := newTestRouter(t, func(w http.ResponseWriter, req *http.Request) { reply(w, "not json") })
	_, err := RunModelTask[out](context.Background(), r, "t", Policy{Models: []string{"a", "b"}}, nil)
	var af *AllModelsFailedError
	if !errors.As(err, &af) || !errors.Is(err, ErrSchema) || len(af.Failures) != 2 {
		t.Fatalf("unexpected error %v", err)
	}
}

func TestFatalAuth(t *testing.T) {
	n := 0
	r := newTestRouter(t, func(w http.ResponseWriter, req *http.Request) {
		n++
		w.WriteHeader(http.StatusUnauthorized)
	})
	_, err := RunModelTask[out](context.Background(), r, "t", Policy{Models: []string{"a", "b"}, MaxRetries: 2}, nil)
	if err == nil || n != 1 {
		t.Fatalf("err=%v calls=%d", err, n)
	}
}
