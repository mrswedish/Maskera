declare module "react" {
  interface HTMLAttributes<T> extends AriaAttributes, DOMAttributes<T> {
    focusgroup?: string;
    focusgroupstart?: string | boolean;
  }
  interface ButtonHTMLAttributes<T> extends HTMLAttributes<T> {
    focusgroupstart?: string | boolean;
  }
}

export {};

